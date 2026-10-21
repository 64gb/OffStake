class_name Employee
extends CharacterBody2D

## Сотрудник офиса

@export var move_speed: float = 100.0

var target_position: Vector2 = Vector2.ZERO
var state: String = "idle"  # idle, moving, working
var move_pattern: String = "xy"  # "xy" или "yx"
var on_reached_callback: Callable = func(): pass

# Для сегментированного движения
var current_segment: int = 0
var path_segments: Array = []

@onready var anim_sprite: AnimatedSprite2D = $AnimatedSprite

func _ready() -> void:
	add_to_group("employees")
	z_index = 4  # Ряд 1 по умолчанию

func _physics_process(delta: float) -> void:
	if state == "moving":
		_move_to_target(delta)
	_update_z_by_position()

# Обновляет z_index на основе Y позиции (глубина офиса)
func _update_z_by_position() -> void:
	var y = position.y
	if y < 315:
		z_index = 4       # Ряд 1: сотрудник (базовый Z=3 + 1)
	elif y < 425:
		z_index = 7       # Ряд 2: сотрудник (базовый Z=6 + 1)
	else:
		z_index = 10      # Ряд 3: сотрудник (базовый Z=9 + 1)

func _move_to_target(_delta: float) -> void:
	# Движение по сегментам: сначала Y, потом X
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
				if on_reached_callback.is_valid():
					on_reached_callback.call()
			return

		velocity = direction * move_speed
		_set_animation("walk")
		move_and_slide()

func _set_animation(anim_name: String) -> void:
	if anim_sprite and anim_sprite.animation != anim_name:
		anim_sprite.play(anim_name)

func go_to(pos: Vector2) -> void:
	target_position = pos
	state = "moving"
	current_segment = 0
	path_segments.clear()

	# Создаём путь: сначала Y, потом X
	# Сегмент 1: движемся по Y до целевой Y
	path_segments.append(Vector2(position.x, pos.y))
	# Сегмент 2: движемся по X до целевой X
	path_segments.append(pos)
