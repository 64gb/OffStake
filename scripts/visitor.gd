class_name Visitor
extends CharacterBody2D

# Типы посетителей
enum VisitorType {
	SIMPLE = 0,    # Простой: паспорт/справка
	MEDIUM = 1,    # Средний: регистрация
	COMPLEX = 2,   # Сложный: визы/юридические
}

# XP награда за каждый тип
const XP_REWARDS = {
	VisitorType.SIMPLE: 3,
	VisitorType.MEDIUM: 5,
	VisitorType.COMPLEX: 10,
}

# VIP посетители (дают больше XP)
const VIP_VISITORS = [36, 37, 38, 40]
const VIP_XP_MULTIPLIER = 2.0  # VIP дают двойной XP

@export var move_speed: float = 60.0
@export var visitor_type: int = 0
@export var visitor_index: int = 1  # 1-40, уникальный номер посетителя

var target_position: Vector2 = Vector2.ZERO
var state: String = "idle"
var move_pattern: String = "xy"
var on_reached_callback: Callable = func(): pass

# Для сегментированного движения
var current_segment: int = 0
var path_segments: Array = []

# Нетерпеливость
var patience_timer: float = 0.0
var max_patience: float = 150.0  # Секунд терпения (устанавливается из main.gd)
var is_impatient: bool = false
var on_impatient_callback: Callable = func(): pass

# Визуал нетерпеливости
var imp_patience_indicator: ColorRect = null

# VIP индикатор (звёздочка)
var vip_indicator: Label = null

# Анимация
@onready var anim_sprite: AnimatedSprite2D = $AnimatedSprite
var sprite_frames: SpriteFrames = null

func _ready() -> void:
	add_to_group("visitors")
	z_index = 6  # Ряд 1: посетитель (базовый Z=3 + 3)

	# Загружаем текстуры и создаём SpriteFrames
	_load_visitor_textures()

	# Индикатор нетерпеливости (красная полоска над головой)
	imp_patience_indicator = ColorRect.new()
	imp_patience_indicator.size = Vector2(20, 4)
	imp_patience_indicator.color = Color(1, 0.3, 0.3, 0.8)
	imp_patience_indicator.visible = false
	add_child(imp_patience_indicator)
	imp_patience_indicator.position = Vector2(-10, -50)

	# VIP индикатор (звёздочка над головой)
	if is_vip():
		vip_indicator = Label.new()
		vip_indicator.text = "⭐"
		vip_indicator.offset_left = -10
		vip_indicator.offset_top = -65
		vip_indicator.offset_right = 10
		vip_indicator.offset_bottom = -48
		vip_indicator.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		vip_indicator.add_theme_font_size_override("font_size", 14)
		vip_indicator.z_index = 10
		add_child(vip_indicator)
		print("[VISITOR] #%d — VIP!" % visitor_index)

func _load_visitor_textures() -> void:
	# Путь к текстурам: res://assets/npc/Посетитель_{N}_{тип}.png
	var base_path = "res://assets/npc/Посетитель_%d_" % visitor_index
	
	var sitting_path = base_path + "сидя.png"
	var left_path = base_path + "левая.png"
	var right_path = base_path + "правая.png"
	
	var sitting_tex = load(sitting_path)
	var left_tex = load(left_path)
	var right_tex = load(right_path)
	
	if not sitting_tex or not left_tex or not right_tex:
		print("[VISITOR] Не удалось загрузить текстуры для посетителя #%d" % visitor_index)
		# Фолбэк — цветной прямоугольник
		return
	
	# Создаём SpriteFrames
	sprite_frames = SpriteFrames.new()
	
	# Анимация "idle" (сидя/стоит) — 1 кадр
	sprite_frames.add_animation("idle")
	sprite_frames.add_frame("idle", sitting_tex, 1.0)
	
	# Анимация "walk" (ходьба) — 2 кадра, медленная ходьба
	sprite_frames.add_animation("walk")
	sprite_frames.add_frame("walk", left_tex, 0.9)
	sprite_frames.add_frame("walk", right_tex, 0.9)

	anim_sprite.sprite_frames = sprite_frames
	anim_sprite.scale = Vector2(0.5, 0.5)
	anim_sprite.play("idle")

func _set_animation(anim_name: String) -> void:
	if anim_sprite and anim_sprite.animation != anim_name:
		anim_sprite.play(anim_name)

func is_vip() -> bool:
	return visitor_index in VIP_VISITORS

func get_xp_reward() -> int:
	var base_xp = XP_REWARDS.get(visitor_type, 3)
	if is_vip():
		return int(base_xp * VIP_XP_MULTIPLIER)
	return base_xp

func _physics_process(delta: float) -> void:
	if state == "moving":
		_move_to_target(delta)
		_update_z_by_position()

	# Таймер терпения работает только в очереди
	if state == "waiting":
		patience_timer += delta
		_update_patience_visual()

		if patience_timer >= max_patience:
			_on_patience_expired()

# Обновляет z_index на основе Y позиции (глубина офиса)
# НЕ меняется когда посетитель сидит в очереди
func _update_z_by_position() -> void:
	if state == "waiting":
		return
	var y = position.y
	if y < 315:
		z_index = 6       # Ряд 1
	elif y < 425:
		z_index = 9       # Ряд 2
	else:
		z_index = 12      # Ряд 3

func _update_patience_visual() -> void:
	if not imp_patience_indicator:
		return

	var remaining = 1.0 - (patience_timer / max_patience)

	# Показываем индикатор когда осталось меньше 50%
	if remaining < 0.5:
		imp_patience_indicator.visible = true
		imp_patience_indicator.size.x = 20.0 * remaining
		# От зелёного к красному
		if remaining < 0.25:
			imp_patience_indicator.color = Color(1, 0.2, 0.2, 0.9)
		else:
			imp_patience_indicator.color = Color(1, 0.7, 0.2, 0.8)
	else:
		imp_patience_indicator.visible = false

func _on_patience_expired() -> void:
	if state == "waiting" and on_impatient_callback.is_valid():
		on_impatient_callback.call()

func _move_to_target(_delta: float) -> void:
	# Движение по сегментам: Y → X → Y (живая траектория)
	if path_segments.size() > 0 and current_segment < path_segments.size():
		var segment_target = path_segments[current_segment]
		var direction = (segment_target - position).normalized()
		var distance = position.distance_to(segment_target)

		if distance < 5.0:
			position = segment_target
			current_segment += 1
			if current_segment >= path_segments.size():
				# Все сегменты пройдены
				state = "idle"
				_set_animation("idle")
				# При остановке — z ниже стула (стул ближе к камере)
				var y = position.y
				if y < 315:
					z_index = 2       # Ряд 1: ниже стула (z=4)
				elif y < 425:
					z_index = 5       # Ряд 2: ниже стула (z=7)
				else:
					z_index = 8       # Ряд 3: ниже стула (z=10)
				if on_reached_callback.is_valid():
					on_reached_callback.call()
			return

		velocity = direction * move_speed
		_set_animation("walk")
		move_and_slide()

func _reached() -> void:
	position = target_position
	state = "idle"
	if on_reached_callback.is_valid():
		on_reached_callback.call()

func go_to_infomat(pos: Vector2) -> void:
	# 🟡 ЖЁЛТЫЙ маршрут: дверь → инфомат
	target_position = pos
	state = "moving"
	current_segment = 0
	path_segments.clear()

	# Сегмент 1: Y вверх до позиции инфомата
	path_segments.append(Vector2(position.x, pos.y))
	# Сегмент 2: X влево до позиции инфомата
	path_segments.append(pos)

func go_to_waiting(pos: Vector2, chair_idx: int = 0) -> void:
	# 🟡 ЖЁЛТЫЙ маршрут: инфомат → зал ожидания
	target_position = pos
	state = "moving"
	current_segment = 0
	path_segments.clear()

	# Ряд 1 (индексы 0-4) и ряд 2 (индексы 5-9) — разные траектории
	if chair_idx < 5:
		# Ряд 1: Влево до X=450, Вниз до Y=450, Потом по Y до стула, Потом по X до стула
		path_segments.append(Vector2(450, position.y))
		path_segments.append(Vector2(450, 450))
		path_segments.append(Vector2(450, pos.y))
		path_segments.append(pos)
	else:
		# Ряд 2: Сразу к стулу по прямой (линейная траектория)
		path_segments.append(pos)

func go_to_window(pos: Vector2) -> void:
	# Посетитель вызвали — сбрасываем таймер нетерпеливости
	patience_timer = 0.0
	if imp_patience_indicator:
		imp_patience_indicator.visible = false

	# 🔵 СИНИЙ маршрут: инфомат/зал → окно
	# Путь: вправо → вверх до ряда → вправо к окну → вниз к окну
	target_position = pos
	state = "moving"
	current_segment = 0
	path_segments.clear()

	# Сегмент 1: X вправо до вертикальной линии (X=650)
	path_segments.append(Vector2(650, position.y))
	# Сегмент 2: Y вверх до уровня ряда окна
	path_segments.append(Vector2(650, pos.y))
	# Сегмент 3: X вправо к позиции окна
	path_segments.append(Vector2(pos.x, pos.y))
	# Сегмент 4: Y вниз к окну (посадка сверху)
	path_segments.append(pos)

func go_to_door(pos: Vector2) -> void:
	# 🟣 ФИОЛЕТОВЫЙ маршрут: инфомат → дверь (уход)
	target_position = pos
	state = "moving"
	current_segment = 0
	path_segments.clear()

	# Сегмент 1: X вправо до позиции двери
	path_segments.append(Vector2(pos.x, position.y))
	# Сегмент 2: Y вниз до позиции двери
	path_segments.append(pos)
