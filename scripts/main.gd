extends Node2D

## Симулятор офиса - исправленная версия

const EMPLOYEE_SCENE = preload("res://scenes/npc/employee.tscn")
const SaveSystem = preload("res://scripts/save_system.gd")
const RandomEvents = preload("res://scripts/random_events.gd")
const Achievements = preload("res://scripts/achievements.gd")
const WINDOW_SCENE = preload("res://scenes/office/service_window.tscn")
const INFOMAT_SCENE = preload("res://scenes/office/infomat.tscn")
const VISITOR_SCENE = preload("res://scenes/npc/visitor.tscn")
const OfficeDecor = preload("res://scripts/office_decor.gd")

# Прекассет текстур декора
const DECOR_TEXTURES = {
	"plant": preload("res://assets/office/Растение_декор.png"),
	"cooler": preload("res://assets/office/Кулер_декор.png"),
	"clock": preload("res://assets/office/часы_декор.png"),
	"trash": preload("res://assets/office/ведро_декор.png"),
	"board": preload("res://assets/office/доска_декор.png"),
}

const CHAIR_WAITING_TEX = preload("res://assets/office/стул посетителя в очереди.png")

# Масштаб для каждого декора (подгонка под размер игры)
const DECOR_SCALES = {
	"plant": Vector2(0.06, 0.06),
	"cooler": Vector2(0.055, 0.055),
	"clock": Vector2(0.045, 0.045),
	"trash": Vector2(0.05, 0.05),
	"board": Vector2(0.06, 0.06),
}

# Позиции
var DOOR_TOP = Vector2(640, 180)
var DOOR_BOTTOM = Vector2(640, 650)
var WINDOW_POS = Vector2(800, 300)
var EMPLOYEE_CHAIR = Vector2(800, 270)
var VISITOR_CHAIR = Vector2(800, 330)
var CABINET_POS = Vector2(1150, 250)  # Шкаф справа (на уровне y=250)
var INFOMAT_POS = Vector2(550, 525)  # Инфомат правее от входа (внутри офиса, не на стене)
var WAITING_START = Vector2(150, 430)  # Зал ожидания слева (не пересекается с окнами)
var WAITING_CHAIR_SPACING: float = 60.0  # Расстояние между стульями
var SAFE_CORRIDOR_Y = 180.0            # Безопасный коридор сверху для траектории

# Состояние
var queue: Array = []
var waiting_chairs: Array[Node] = []
var windows: Array[ServiceWindow] = []  # Массив окон приёма

# Прогресс
var xp: int = 100000  # Начальное XP
var served_count: int = 0
var reputation: int = 50  # Репутация (0-100)
var left_count: int = 0  # Сколько посетителей ушло

# Система приёма (5 уровней)
var system_level: int = 0  # Текущий уровень (0-4)
var system_levels: Array = [
	{"name": "Начальная", "time": 180.0, "cost": 0},
	{"name": "Улучшенная", "time": 150.0, "cost": 150},
	{"name": "Автоматическая", "time": 120.0, "cost": 400},
	{"name": "Виртуальный помощник", "time": 100.0, "cost": 1000},
	{"name": "Продвинутая АИС", "time": 60.0, "cost": 2500}
]

# Улучшения системы (качество - 5 ячеек по -1 сек)
var system_quality: Array[bool] = [false, false, false, false, false]
var system_quality_costs: Array[int] = [3, 6, 12, 24, 48]

# Улучшения системы (обновление - 5 ячеек по -5 сек)
var system_upgrade: Array[bool] = [false, false, false, false, false]
var system_upgrade_costs: Array[int] = [10, 20, 40, 60, 80]

# Бусты (временные ускорения)
var active_boosts: Dictionary = {}  # {boost_id: time_remaining}
var boost_definitions: Dictionary = {
	"coffee": {"name": "Кофе", "cost": 20, "multiplier": 0.8, "duration": 60.0},
	"energy": {"name": "Энергетик", "cost": 40, "multiplier": 0.6, "duration": 120.0},
	"training": {"name": "Обучение", "cost": 50, "multiplier": 0.5, "duration": 180.0},
	"praise": {"name": "Похвала", "cost": 70, "multiplier": 0.25, "duration": 300.0},
	"promotion": {"name": "Повышение", "cost": 100, "multiplier": 0.0, "duration": 420.0}
}

# Очередь
var max_queue_size: int = 2  # Максимум мест в очереди
var queue_upgrades: Array = [
	{"chairs": 2, "cost": 0, "purchased": true},
	{"chairs": 3, "cost": 100, "purchased": false},
	{"chairs": 5, "cost": 400, "purchased": false},
	{"chairs": 7, "cost": 800, "purchased": false},
	{"chairs": 10, "cost": 1500, "purchased": false}
]

var visitor_spawn_timer: float = 3.0

# Сотрудник
var employee_node: Node = null  # Первый сотрудник
var employee_at_desk: bool = true  # true = за столом, false = у шкафа
var employees_busy: Dictionary = {}  # {employee: true/false} - кто занят

# Новые сотрудники (ШАГ 1)
var employees: Array = []  # Массив всех сотрудников
var employee_windows: Dictionary = {}  # {window: employee} - привязка
var all_employee_chairs: Array = []    #预生成的 стулья сотрудников
var all_visitor_chairs: Array = []     #预生成的 стулья посетителей
var max_employees: int = 9  # Максимум сотрудников (сетка 3x3)
var employee_cost: int = 150  # Цена найма (растёт на 40%)
var employee_positions: Array = [
	Vector2(750, 260), Vector2(900, 260), Vector2(1050, 260),
	Vector2(750, 370), Vector2(900, 370), Vector2(1050, 370),
	Vector2(750, 480), Vector2(900, 480), Vector2(1050, 480)
]

# Промежуточные точки для траекторий сотрудников (от двери к столу)
var employee_spawn_paths: Array = [
	Vector2(680, 260),  # Стол 1: дверь → (680,260) → (750,260)
	Vector2(750, 200),  # Стол 2: дверь → (750,200) → (900,260)
	Vector2(850, 180),  # Стол 3: дверь → (850,180) → (1050,260)
	Vector2(680, 370),  # Стол 4: дверь → (680,370) → (750,370)
	Vector2(750, 310),  # Стол 5: дверь → (750,310) → (900,370)
	Vector2(850, 260),  # Стол 6: дверь → (850,260) → (1050,370)
	Vector2(680, 480),  # Стол 7: дверь → (680,480) → (750,480)
	Vector2(750, 420),  # Стол 8: дверь → (750,420) → (900,480)
	Vector2(850, 370),  # Стол 9: дверь → (850,370) → (1050,480)
]

# Отладка ШАГА 1
func _print_employee_vars() -> void:
	print("=== ШАГ 1: Переменные сотрудников ===")
	print("  max_employees: ", max_employees)
	print("  employee_cost: ", employee_cost)
	print("  employee_positions.size(): ", employee_positions.size())
	print("  employees.size(): ", employees.size())
	print("  employee_windows.size(): ", employee_windows.size())
	print("====================================")

# Инициализация всех рабочих мест (предварительная генерация)
func _init_all_workplaces() -> void:
	print("[ИНИЦИАЛИЗАЦИЯ] Создаём ", employee_positions.size(), " рабочих мест...")
	for i in range(employee_positions.size()):
		var pos = employee_positions[i]
		var row = i / 3  # 0, 1, 2 — ряд
		var row_z = 3 + row * 3  # Ряд 1: z=3, ряд 2: z=6, ряд 3: z=9

		# Окно
		var window = WINDOW_SCENE.instantiate()
		window.position = pos
		window.z_index = row_z
		window.visible = (i == 0)  # Видно только первое
		add_child(window)
		windows.append(window)

		# Стул сотрудника
		var emp_chair = ColorRect.new()
		emp_chair.position = pos + Vector2(0, 30)
		emp_chair.offset_left = -12; emp_chair.offset_top = -10
		emp_chair.offset_right = 12; emp_chair.offset_bottom = 10
		emp_chair.color = Color(0.3, 0.3, 0.3, 1)
		emp_chair.z_index = row_z
		emp_chair.visible = (i == 0)
		add_child(emp_chair)
		all_employee_chairs.append(emp_chair)

		# Стул посетителя
		var vis_chair = ColorRect.new()
		vis_chair.position = pos + Vector2(0, -45)
		vis_chair.offset_left = -10; vis_chair.offset_top = -10
		vis_chair.offset_right = 10; vis_chair.offset_bottom = 10
		vis_chair.color = Color(0.4, 0.4, 0.4, 1)
		vis_chair.z_index = row_z + 1
		vis_chair.visible = (i == 0)
		add_child(vis_chair)
		all_visitor_chairs.append(vis_chair)
		
	print("[ИНИЦИАЛИЗАЦИЯ] Готово. Активно окно #1")

# Активация рабочего места при найме сотрудника
func _activate_workplace(index: int) -> void:
	if index < windows.size():
		windows[index].visible = true
		all_employee_chairs[index].visible = true
		all_visitor_chairs[index].visible = true
		windows[index].z_index = 3 + (index / 3) * 3
		all_employee_chairs[index].z_index = windows[index].z_index
		all_visitor_chairs[index].z_index = windows[index].z_index
		print("[ШАГ 4] Рабочее место #", index + 1, " активировано (z=", windows[index].z_index, ")")

# Находим слоты декора из сцены
func _find_decor_slots() -> void:
	var container = get_node_or_null("DecorSlots")
	if not container:
		print("[ДЕКОР] Контейнер DecorSlots не найден!")
		return
	
	for child in container.get_children():
		var decor_id = child.get("decor_id")
		if decor_id and decor_id != "":
			decor_slots[decor_id] = child
			print("[ДЕКОР] Найден слот: ", decor_id, " на ", child.position)

# Обновление отображения слотов декора
func _update_decor_slots_display() -> void:
	for decor_id in decor_slots.keys():
		var slot = decor_slots[decor_id]
		var purchased = office_decor_system.is_purchased(decor_id)
		
		var indicator = slot.get_node_or_null("SlotIndicator")
		var label = slot.get_node_or_null("SlotLabel")
		
		if purchased:
			# Скрываем "?" и фон слота
			if indicator:
				indicator.visible = false
			if label:
				label.visible = false

			# Удаляем старый декор если есть
			var old_decor = slot.get_node_or_null("DecorVisual")
			if old_decor:
				old_decor.queue_free()

			# Рисуем декор — Sprite2D с текстурой
			var decor_def = office_decor_system.decor_definitions[decor_id]
			var texture = DECOR_TEXTURES.get(decor_id)
			if texture:
				var visual = Sprite2D.new()
				visual.name = "DecorVisual"
				visual.texture = texture
				visual.scale = DECOR_SCALES.get(decor_id, Vector2(0.05, 0.05))
				visual.z_index = 3
				slot.add_child(visual)

				print("[ДЕКОР] Отображён: ", decor_def.name)
			else:
				print("[ДЕКОР] Текстура не найдена для: ", decor_id)
		else:
			# Показываем "?" слот
			if indicator:
				indicator.visible = true
			if label:
				label.visible = true

# Отображение всего купленного декора (вызывается после загрузки сохранения)
func _show_all_purchased_decor() -> void:
	_update_decor_slots_display()

# Покупка декора
func _on_purchase_decor(decor_id: String) -> void:
	print("[ДЕКОР] Покупка: ", decor_id)
	var result = office_decor_system.purchase(decor_id, xp)
	
	if result.success:
		xp -= result.xp_cost
		print("[ДЕКОР] ", result.message, " (-", result.xp_cost, " XP)")
		_update_decor_slots_display()
		_update_ui()
		_update_progress_panel()
		_save_progress()
	else:
		print("[ДЕКОР] Ошибка: ", result.message)

# Пауза
var game_paused: bool = false

# Таймер для обновления UI бустов
var boost_ui_timer: float = 0.0

# Случайные события
var random_events_system = RandomEvents.new()
var active_event_notification: Label = null

# Достижения
var achievements_system = Achievements.new()

# Декор офиса
var office_decor_system = OfficeDecor.new()
var decor_slots: Dictionary = {}  # {decor_id: Node} — слоты из сцены

func _ready() -> void:
	# Инфомат
	var infomat = INFOMAT_SCENE.instantiate()
	add_child(infomat)
	infomat.position = INFOMAT_POS  # (550, 520) - внутри офиса

	# Узел для уведомлений о событиях
	active_event_notification = Label.new()
	active_event_notification.offset_left = 110
	active_event_notification.offset_top = 225
	active_event_notification.offset_right = 450
	active_event_notification.offset_bottom = 290
	active_event_notification.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	active_event_notification.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	active_event_notification.add_theme_font_size_override("font_size", 13)
	active_event_notification.add_theme_color_override("font_color", Color(1, 1, 0.5, 1))
	active_event_notification.visible = false
	add_child(active_event_notification)

	# Стулья ожидания (сразу 10, но видны только max_queue_size)
	_create_waiting_chairs(10)

	# Инициализируем все 10 рабочих мест (скрыты, кроме первого)
	_init_all_workplaces()

	# Находим слоты декора из сцены
	_find_decor_slots()

	# Создаём первого сотрудника и привязываем к первому окну
	employee_node = EMPLOYEE_SCENE.instantiate()
	if employee_node:
		add_child(employee_node)
		employee_node.position = DOOR_TOP
		employees.append(employee_node)
		employee_windows[windows[0]] = employee_node
		employee_node.set_meta("workplace_pos", employee_positions[0])

		# Первый сотрудник идёт через промежуточную точку
		var via_point = employee_spawn_paths[0]
		employee_node.go_to(via_point)
		employee_node.on_reached_callback = func():
			employee_node.go_to(employee_positions[0])
			employee_node.on_reached_callback = func():
				employee_at_desk = true
		
		await get_tree().create_timer(1.0).timeout
	else:
		print("  ОШИБКА: employee_node не создан!")

	# Ждём пока UI загрузится (UI уже есть в main.tscn как вложенная сцена)
	await get_tree().create_timer(0.2).timeout

	# Подключаем кнопки
	_connect_buttons()

	# Скрываем меню паузы
	var pause_menu = get_node_or_null("UI/PauseMenu")
	if pause_menu:
		pause_menu.visible = false

	# Отладка ШАГА 1
	_print_employee_vars()

	# Создаём вкладки для TabContainer (в Godot 4 они не создаются из сцены)
	_create_tabs()

	# Обновляем UI
	_update_ui()
	_update_waiting_chairs_visibility()  # Показываем только 2 стула изначально

	# Загружаем сохранение если есть
	_load_progress()
	
	# Показываем купленный декор (если нет сохранения — ничего не покажет)
	_show_all_purchased_decor()

func _create_tabs() -> void:
	var ui = get_node_or_null("UI")
	if not ui:
		return

	var tabs = ui.get_node_or_null("ProgressPanel/Tabs")
	if not tabs:
		return

	# Удаляем старые вкладки если есть
	for child in tabs.get_children():
		child.queue_free()

	# Создаём 5 вкладок через ScrollContainer
	var tab_names = ["SystemTab", "QueueTab", "BoostsTab", "EmployeesTab", "DecorTab"]
	var tab_titles = ["Система", "Очередь", "Бусты", "Сотрудники", "Декор"]

	for i in range(tab_names.size()):
		var scroll_tab = ScrollContainer.new()
		scroll_tab.name = tab_names[i]
		tabs.add_child(scroll_tab)
		tabs.set_tab_title(i, tab_titles[i])
		
		# Добавляем Content внутрь
		var content = VBoxContainer.new()
		content.name = "Content"
		content.offset_left = 0
		content.offset_top = 0
		content.offset_right = 480
		content.offset_bottom = 600
		content.add_theme_constant_override("separation", 10)
		scroll_tab.add_child(content)

	# Подключаем сигналы
	var system_tab = tabs.get_node("SystemTab")
	var queue_tab = tabs.get_node("QueueTab")
	var boosts_tab = tabs.get_node("BoostsTab")
	_connect_tab_signals(system_tab, queue_tab, boosts_tab)

func _connect_tab_signals(system_tab: Node, queue_tab: Node, boosts_tab: Node) -> void:
	# Подключаем сигналы системы
	if system_tab.has_signal("purchase_quality"):
		system_tab.connect("purchase_quality", _on_purchase_quality)
	if system_tab.has_signal("purchase_upgrade"):
		system_tab.connect("purchase_upgrade", _on_purchase_upgrade)
	if system_tab.has_signal("upgrade_system"):
		system_tab.connect("upgrade_system", _on_upgrade_system)
	
	# Подключаем сигналы очереди
	if queue_tab.has_signal("upgrade_queue"):
		queue_tab.connect("upgrade_queue", _on_upgrade_queue)
	
	# Подключаем сигналы бустов
	if boosts_tab.has_signal("activate_boost"):
		boosts_tab.connect("activate_boost", _on_activate_boost)

func _connect_buttons() -> void:
	# UI теперь вложенная сцена в main.tscn
	var ui = get_node_or_null("UI")
	if not ui:
		print("ОШИБКА: UI не найден! Проверьте что ui.tscn добавлен в main.tscn")
		return
	
	# Включаем PROCESS_MODE_ALWAYS чтобы UI работал на паузе
	ui.process_mode = CanvasLayer.PROCESS_MODE_ALWAYS
	
	var pause_btn = ui.get_node_or_null("PauseButton")
	if pause_btn:
		pause_btn.pressed.connect(_on_pause_pressed)
		print("✓ PauseButton подключена")
	else:
		print("✗ PauseButton НЕ найдена!")

	var menu_btn = ui.get_node_or_null("MenuButton")
	if menu_btn:
		menu_btn.pressed.connect(_on_menu_pressed)
		print("✓ MenuButton подключена")
	else:
		print("✗ MenuButton НЕ найдена!")
	
	var progress_btn = ui.get_node_or_null("ProgressButton")
	if progress_btn:
		progress_btn.pressed.connect(_on_progress_pressed)
		print("✓ ProgressButton подключена")
	else:
		print("✗ ProgressButton НЕ найдена!")

	var pause_menu = ui.get_node_or_null("PauseMenu")
	if pause_menu:
		print("✓ PauseMenu найден, детей: ", pause_menu.get_child_count())

		var resume_btn = pause_menu.get_node_or_null("ResumeButton")
		if resume_btn:
			resume_btn.pressed.connect(_on_resume_pressed)
			print("✓ ResumeButton подключена")
		else:
			print("✗ ResumeButton НЕ найдена!")

		var exit_btn = pause_menu.get_node_or_null("ExitButton")
		if exit_btn:
			exit_btn.pressed.connect(_on_exit_pressed)
			print("✓ ExitButton подключена")
		else:
			print("✗ ExitButton НЕ найдена!")

		var delete_save_btn = pause_menu.get_node_or_null("DeleteSaveButton")
		if delete_save_btn:
			delete_save_btn.pressed.connect(_on_delete_save_pressed)
			print("✓ DeleteSaveButton подключена")
		else:
			print("✗ DeleteSaveButton НЕ найдена!")
	else:
		print("✗ PauseMenu НЕ найден!")
	
	# Подключаем кнопку закрытия панели прогресса
	var progress_panel = ui.get_node_or_null("ProgressPanel")
	if progress_panel:
		var close_btn = progress_panel.get_node_or_null("CloseButton")
		if close_btn:
			close_btn.pressed.connect(_on_close_progress)
			print("✓ CloseButton подключена")
		else:
			print("✗ CloseButton НЕ найдена!")
	else:
		print("✗ ProgressPanel НЕ найден!")

func _on_pause_pressed() -> void:
	game_paused = not game_paused
	get_tree().paused = game_paused
	
	# Меню НЕ показываем при паузе
	var pause_menu = get_node_or_null("UI/PauseMenu")
	if pause_menu:
		pause_menu.visible = false
	
	# Меняем текст кнопки
	var pause_btn = get_node_or_null("UI/PauseButton")
	if pause_btn:
		pause_btn.text = "▶" if game_paused else "⏸"
	
	print("Пауза: ", game_paused)

func _on_menu_pressed() -> void:
	print("☰ Меню нажато!")
	game_paused = true
	get_tree().paused = true

	# Показываем меню
	var pause_menu = get_node_or_null("UI/PauseMenu")
	if pause_menu:
		pause_menu.visible = true
		print("  Меню показано")

	# Кнопка паузы показывает "играть"
	var pause_btn = get_node_or_null("UI/PauseButton")
	if pause_btn:
		pause_btn.text = "▶"

func _on_resume_pressed() -> void:
	print("Продолжить нажато!")
	game_paused = false
	get_tree().paused = false

	# Скрываем меню
	var pause_menu = get_node_or_null("UI/PauseMenu")
	if pause_menu:
		pause_menu.visible = false
		print("  Меню скрыто")

	# Кнопка паузы показывает "пауза"
	var pause_btn = get_node_or_null("UI/PauseButton")
	if pause_btn:
		pause_btn.text = "⏸"

func _on_exit_pressed() -> void:
	print("Выйти нажато! Выход из игры...")
	_save_progress()
	get_tree().quit()

func _on_delete_save_pressed() -> void:
	print("🗑 Удалить сохранение нажато!")
	if SaveSystem.delete_save():
		print("  Сохранение удалено. Перезапуск...")
		# Сбрасываем данные
		SaveSystem.reset_data()
		# Перезагружаем сцену
		get_tree().reload_current_scene()
	else:
		print("  Сохранение не найдено!")

func _on_progress_pressed() -> void:
	print("📊 Прогресс нажато!")
	game_paused = true
	get_tree().paused = true
	
	# Показываем панель прогресса
	var progress_panel = get_node_or_null("UI/ProgressPanel")
	if progress_panel:
		progress_panel.visible = true
		print("  Панель прогресса показана")
		_update_progress_panel()

func _on_close_progress() -> void:
	print("Закрыть прогресс нажато!")
	game_paused = false
	get_tree().paused = false
	
	# Скрываем панель прогресса
	var progress_panel = get_node_or_null("UI/ProgressPanel")
	if progress_panel:
		progress_panel.visible = false
		print("  Панель прогресса скрыта")

func _update_progress_panel() -> void:
	# Обновляем содержимое панели прогресса
	print("  XP: %d, Уровень системы: %d" % [xp, system_level])

	var ui = get_node_or_null("UI")
	if not ui:
		return

	# Обновляем вкладки через их update_display()
	_fill_system_tab(ui)
	_fill_queue_tab(ui)
	_fill_boosts_tab(ui)
	_fill_employees_tab(ui)
	_fill_decor_tab(ui)

func _fill_system_tab(ui: Node) -> void:
	var scroll_tab = ui.get_node_or_null("ProgressPanel/Tabs/SystemTab")
	if not scroll_tab:
		print("  ОШИБКА: SystemTab не найдена!")
		return

	# Находим Content внутри ScrollContainer
	var content = scroll_tab.get_node_or_null("Content")
	if not content:
		print("  ОШИБКА: Content не найден!")
		return

	# Очищаем содержимое
	for child in content.get_children():
		child.queue_free()

	# Панель бонусов системы (сверху)
	var service_time = get_service_time()
	var quality_count = 0
	var upgrade_count = 0
	for v in system_quality:
		if v:
			quality_count += 1
	for v in system_upgrade:
		if v:
			upgrade_count += 1
	
	var bonus_parts: PackedStringArray = []
	bonus_parts.append("⏱ %.0f сек" % service_time)
	if quality_count > 0:
		bonus_parts.append("🔧 -%d сек" % quality_count)
	if upgrade_count > 0:
		bonus_parts.append("🔄 -%d сек" % (upgrade_count * 5))
	if bonus_parts.size() <= 1 and quality_count == 0 and upgrade_count == 0:
		bonus_parts.append("📦 без улучшений")
	
	var bonus_panel = Panel.new()
	bonus_panel.custom_minimum_size = Vector2(480, 30)
	content.add_child(bonus_panel)
	
	var bonus_label = Label.new()
	bonus_label.offset_left = 10
	bonus_label.offset_top = 3
	bonus_label.offset_right = 470
	bonus_label.offset_bottom = 28
	bonus_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	bonus_label.text = " | ".join(bonus_parts)
	bonus_label.add_theme_font_size_override("font_size", 12)
	bonus_label.add_theme_color_override("font_color", Color(0.7, 0.9, 0.7, 1))
	bonus_panel.add_child(bonus_label)

	# Создаём панель для каждой системы
	for i in range(system_levels.size()):
		var level_data = system_levels[i]
		var is_purchased = (i <= system_level)
		var is_current = (i == system_level)
		
		# Панель уровня
		var level_panel = Panel.new()
		level_panel.custom_minimum_size = Vector2(480, 35)
		content.add_child(level_panel)
		
		# Кнопка уровня
		var level_btn = Button.new()
		level_btn.offset_left = 10
		level_btn.offset_top = 5
		level_btn.offset_right = 470
		level_btn.offset_bottom = 30
		
		if i < system_level:
			level_btn.text = "✓ %s" % level_data.name
			level_btn.disabled = true
		elif is_current:
			level_btn.text = "▶ %s (Текущая)" % level_data.name
			level_btn.disabled = true
		else:
			level_btn.text = "▶ %s - %d XP" % [level_data.name, level_data.cost]
			if xp < level_data.cost:
				level_btn.disabled = true
			else:
				level_btn.pressed.connect(_on_upgrade_system.bind(i))
		
		level_panel.add_child(level_btn)
		
		# Если это текущий уровень — добавляем улучшения
		if is_current:
			# Панель улучшений
			var upgrade_panel = Panel.new()
			upgrade_panel.custom_minimum_size = Vector2(480, 100)
			content.add_child(upgrade_panel)
			
			var upgrade_container = VBoxContainer.new()
			upgrade_container.offset_left = 15
			upgrade_container.offset_top = 10
			upgrade_container.offset_right = -15
			upgrade_container.offset_bottom = -10
			upgrade_panel.add_child(upgrade_container)
			
			# Заголовок панели улучшений
			var panel_header = Label.new()
			panel_header.text = "⚙ Улучшения системы"
			panel_header.offset_left = 15
			panel_header.offset_top = 0
			panel_header.offset_right = 465
			panel_header.offset_bottom = 20
			panel_header.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
			panel_header.add_theme_font_size_override("font_size", 14)
			upgrade_container.add_child(panel_header)
			
			# Качество
			var quality_row = HBoxContainer.new()
			quality_row.custom_minimum_size = Vector2(450, 32)
			quality_row.add_theme_constant_override("separation", 10)
			
			var q_label = Label.new()
			q_label.text = "Качество:"
			q_label.custom_minimum_size = Vector2(70, 30)
			q_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
			quality_row.add_child(q_label)
			
			var q_slots = HBoxContainer.new()
			q_slots.add_theme_constant_override("separation", 5)
			for slot in range(5):
				var indicator = ColorRect.new()
				indicator.custom_minimum_size = Vector2(20, 20)
				if system_quality[slot]:
					indicator.color = Color(0.2, 0.9, 0.2, 1)
				else:
					indicator.color = Color(0.4, 0.4, 0.4, 1)
				q_slots.add_child(indicator)
			quality_row.add_child(q_slots)
			
			var q_buy_btn = Button.new()
			q_buy_btn.custom_minimum_size = Vector2(140, 30)
			var q_cost_mult = pow(2, system_level)
			var next_slot = _get_next_available_slot(system_quality)
			if next_slot >= 0:
				var cost = system_quality_costs[next_slot] * q_cost_mult
				q_buy_btn.text = "🛒 Купить за %d XP" % cost
				if xp < cost:
					q_buy_btn.disabled = true
				else:
					q_buy_btn.pressed.connect(_on_purchase_quality.bind(next_slot))
			else:
				q_buy_btn.text = "MAX"
				q_buy_btn.disabled = true
			quality_row.add_child(q_buy_btn)
			
			upgrade_container.add_child(quality_row)
			
			# Обновление
			var upgrade_row = HBoxContainer.new()
			upgrade_row.custom_minimum_size = Vector2(450, 32)
			upgrade_row.add_theme_constant_override("separation", 10)
			
			var u_label = Label.new()
			u_label.text = "Обновление:"
			u_label.custom_minimum_size = Vector2(70, 30)
			u_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
			upgrade_row.add_child(u_label)
			
			var u_slots = HBoxContainer.new()
			u_slots.add_theme_constant_override("separation", 5)
			for slot in range(5):
				var indicator = ColorRect.new()
				indicator.custom_minimum_size = Vector2(20, 20)
				if system_upgrade[slot]:
					indicator.color = Color(0.2, 0.9, 0.2, 1)
				else:
					indicator.color = Color(0.4, 0.4, 0.4, 1)
				u_slots.add_child(indicator)
			upgrade_row.add_child(u_slots)
			
			var u_buy_btn = Button.new()
			u_buy_btn.custom_minimum_size = Vector2(140, 30)
			next_slot = _get_next_available_slot(system_upgrade)
			if next_slot >= 0:
				var cost = system_upgrade_costs[next_slot] * pow(2, system_level)
				u_buy_btn.text = "🛒 Купить за %d XP" % cost
				if xp < cost:
					u_buy_btn.disabled = true
				else:
					u_buy_btn.pressed.connect(_on_purchase_upgrade.bind(next_slot))
			else:
				u_buy_btn.text = "MAX"
				u_buy_btn.disabled = true
			upgrade_row.add_child(u_buy_btn)
			
			upgrade_container.add_child(upgrade_row)

func _fill_queue_tab(ui: Node) -> void:
	var scroll_tab = ui.get_node_or_null("ProgressPanel/Tabs/QueueTab")
	if not scroll_tab:
		return

	var content = scroll_tab.get_node_or_null("Content")
	if not content:
		return

	# Очищаем
	for child in content.get_children():
		child.queue_free()

	# Текущее количество мест
	var label = Label.new()
	label.text = "Количество мест: %d" % max_queue_size
	label.offset_left = 10
	label.offset_top = 5
	label.offset_right = 470
	label.offset_bottom = 30
	label.add_theme_font_size_override("font_size", 14)
	content.add_child(label)

	# Кнопки улучшения (по ТЗ: 2, 3, 5, 7, 10 стульев)
	var y_offset = 40
	
	# Находим индекс последнего купленного улучшения
	var last_purchased_index = -1
	for i in range(queue_upgrades.size() - 1, 0, -1):
		if queue_upgrades[i].purchased:
			last_purchased_index = i
			break
	
	for i in range(1, queue_upgrades.size()):
		var upgrade = queue_upgrades[i]

		var btn = Button.new()
		btn.offset_left = 10
		btn.offset_top = y_offset
		btn.offset_right = 470
		btn.offset_bottom = y_offset + 35

		if upgrade.purchased:
			if i == last_purchased_index:
				# Последнее купленное — показываем галочку
				btn.text = "%d стульев ✓" % upgrade.chairs
			else:
				# Куплено, но не последнее — без галочки
				btn.text = "%d стульев" % upgrade.chairs
			btn.disabled = true
		elif upgrade.chairs <= max_queue_size:
			# Нельзя купить меньшее количество
			btn.text = "%d стульев" % upgrade.chairs
			btn.disabled = true
		else:
			btn.text = "%d стульев - %d XP" % [upgrade.chairs, upgrade.cost]
			if xp < upgrade.cost:
				btn.disabled = true
			else:
				btn.pressed.connect(_on_upgrade_queue.bind(i))

		content.add_child(btn)
		y_offset += 40

func _fill_boosts_tab(ui: Node) -> void:
	var scroll_tab = ui.get_node_or_null("ProgressPanel/Tabs/BoostsTab")
	if not scroll_tab:
		return

	var content = scroll_tab.get_node_or_null("Content")
	if not content:
		return

	# Очищаем
	for child in content.get_children():
		child.queue_free()

	# Панель активных бустов (сверху)
	var active_bonus_panel = Panel.new()
	active_bonus_panel.custom_minimum_size = Vector2(480, 35)
	content.add_child(active_bonus_panel)
	
	var active_label = Label.new()
	active_label.offset_left = 10
	active_label.offset_top = 5
	active_label.offset_right = 470
	active_label.offset_bottom = 30
	active_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	
	if active_boosts.size() > 0:
		var boost_lines: PackedStringArray = []
		var total_mult = 1.0
		for bid in active_boosts.keys():
			var b = boost_definitions[bid]
			boost_lines.append("%s: %.0f сек" % [b.name, active_boosts[bid]])
			total_mult *= b.multiplier
		active_label.text = "⚡ АКТИВНЫ: " + " | ".join(boost_lines) + "\nМножитель: x%.2f" % (1.0 - total_mult)
		active_label.add_theme_color_override("font_color", Color(1, 0.9, 0.3, 1))
	else:
		active_label.text = "⚡ Нет активных бустов"
		active_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7, 1))
	
	active_label.add_theme_font_size_override("font_size", 12)
	active_bonus_panel.add_child(active_label)

	# Создаём кнопки для каждого буста
	var y_offset = 45
	for boost_id in boost_definitions.keys():
		var boost = boost_definitions[boost_id]

		var btn = Button.new()
		btn.offset_left = 10
		btn.offset_top = y_offset
		btn.offset_right = 470
		btn.offset_bottom = y_offset + 35

		# Проверяем активен ли буст
		var is_active = boost_id in active_boosts
		if is_active:
			btn.text = "%s ✓ (%d сек)" % [boost.name, int(active_boosts[boost_id])]
			btn.disabled = true
		else:
			btn.text = "%s - %d XP (%d сек)" % [boost.name, boost.cost, int(boost.duration)]
			if xp < boost.cost:
				btn.disabled = true
			else:
				btn.pressed.connect(_on_activate_boost.bind(boost_id))

		content.add_child(btn)
		y_offset += 40

func _fill_employees_tab(ui: Node) -> void:
	var scroll_tab = ui.get_node_or_null("ProgressPanel/Tabs/EmployeesTab")
	if not scroll_tab:
		return

	var content = scroll_tab.get_node_or_null("Content")
	if not content:
		return

	# Очищаем содержимое
	for child in content.get_children():
		child.queue_free()

	# ШАГ 2: Заголовок с количеством сотрудников
	var label = Label.new()
	label.text = "Нанято сотрудников: %d/%d" % [employees.size(), max_employees]
	label.offset_left = 10
	label.offset_top = 5
	label.offset_right = 470
	label.offset_bottom = 30
	label.add_theme_font_size_override("font_size", 14)
	content.add_child(label)
	print("[ШАГ 2] Заголовок создан: ", label.text)

	# ШАГ 2: Кнопка найма
	if employees.size() < max_employees:
		var hire_btn = Button.new()
		hire_btn.offset_left = 10
		hire_btn.offset_top = 40
		hire_btn.offset_right = 470
		hire_btn.offset_bottom = 75
		hire_btn.text = "👥 Нанять сотрудника - %d XP" % employee_cost
		
		if xp < employee_cost:
			hire_btn.disabled = true
			print("[ШАГ 2] Кнопка найма создана (disabled - не хватает XP)")
		else:
			hire_btn.pressed.connect(_on_hire_employee)
			print("[ШАГ 2] Кнопка найма создана (active)")
		
		content.add_child(hire_btn)
	else:
		var max_label = Label.new()
		max_label.text = "✓ Максимальное количество сотрудников"
		max_label.offset_left = 10
		max_label.offset_top = 40
		max_label.offset_right = 470
		max_label.offset_bottom = 70
		content.add_child(max_label)
		print("[ШАГ 2] Достигнут максимум сотрудников")

func _fill_decor_tab(ui: Node) -> void:
	var scroll_tab = ui.get_node_or_null("ProgressPanel/Tabs/DecorTab")
	if not scroll_tab:
		return

	var content = scroll_tab.get_node_or_null("Content")
	if not content:
		return

	# Очищаем содержимое
	for child in content.get_children():
		child.queue_free()

	# Панель бонусов декора (сверху)
	var decor_bonus_panel = Panel.new()
	decor_bonus_panel.custom_minimum_size = Vector2(480, 35)
	content.add_child(decor_bonus_panel)
	
	var bonus_label = Label.new()
	bonus_label.offset_left = 10
	bonus_label.offset_top = 5
	bonus_label.offset_right = 470
	bonus_label.offset_bottom = 30
	bonus_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	
	var bonus_text = office_decor_system.get_bonus_description()
	if bonus_text != "":
		bonus_label.text = "🏢 Бонусы: " + bonus_text
		bonus_label.add_theme_color_override("font_color", Color(0.7, 0.9, 0.7, 1))
	else:
		bonus_label.text = "🏢 Купите декор для бонусов"
		bonus_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7, 1))
	bonus_label.add_theme_font_size_override("font_size", 12)
	decor_bonus_panel.add_child(bonus_label)

	# Заголовок
	var label = Label.new()
	label.text = "🏢 Декор офиса"
	label.offset_left = 10
	label.offset_top = 5
	label.offset_right = 470
	label.offset_bottom = 30
	label.add_theme_font_size_override("font_size", 16)
	content.add_child(label)

	# Список декора
	var y_offset = 75
	for decor_id in office_decor_system.get_all_decor_ids():
		var decor = office_decor_system.decor_definitions[decor_id]
		var purchased = office_decor_system.is_purchased(decor_id)

		var btn = Button.new()
		btn.offset_left = 10
		btn.offset_top = y_offset
		btn.offset_right = 470
		btn.offset_bottom = y_offset + 35

		if purchased:
			btn.text = "%s ✓" % decor.name
			btn.disabled = true
		else:
			btn.text = "%s - %d XP" % [decor.name, decor.cost]
			if xp < decor.cost:
				btn.disabled = true
			else:
				btn.pressed.connect(_on_purchase_decor.bind(decor_id))

		content.add_child(btn)
		y_offset += 40

func _get_next_available_slot(array: Array) -> int:
	# Находим первую некупленную ячейку
	for i in range(array.size()):
		if not array[i]:
			return i
	return -1  # Все куплены

# ШАГ 3: Функция найма сотрудника
func _on_hire_employee() -> void:
	print("[ШАГ 3] Кнопка найма нажата!")
	
	# Проверка 1: Не достигнут ли максимум
	if employees.size() >= max_employees:
		print("  [ШАГ 3] Максимум сотрудников достигнуто!")
		return
	
	# Проверка 2: Хватает ли XP
	if xp < employee_cost:
		print("  [ШАГ 3] Недостаточно XP! Нужно ", employee_cost, " есть ", xp)
		return
	
	# Списание XP
	xp -= employee_cost
	print("  [ШАГ 3] Списано ", employee_cost, " XP, осталось ", xp)
	
	# Увеличение цены на 40%
	employee_cost = int(employee_cost * 1.4)
	print("  [ШАГ 3] Новая цена найма: ", employee_cost)
	
	# Увеличение счётчика
	var new_employee_index = employees.size()
	print("  [ШАГ 3] Нанимаем сотрудника номер ", new_employee_index + 1)
	
	# Активируем预生成的 рабочее место
	_activate_workplace(new_employee_index)
	
	# Спавним сотрудника
	_spawn_employee(employee_positions[new_employee_index], windows[new_employee_index])
	
	# Обновляем UI
	_update_ui()
	_update_progress_panel()
	_save_progress()

	# Проверяем достижения
	_check_achievements()

	print("  [ШАГ 3] Сотрудник ", new_employee_index + 1, " нанят!")

# ШАГ 5: Спавн сотрудника (с уникальной траекторией)
func _spawn_employee(workplace_pos: Vector2, window: ServiceWindow) -> void:
	print("[ШАГ 5] Спавн сотрудника...")

	var employee = EMPLOYEE_SCENE.instantiate()
	if not employee:
		print("  [ШАГ 5] ОШИБКА: Не удалось создать сотрудника!")
		return

	add_child(employee)
	employee.position = DOOR_TOP  # Спавн у верхней двери
	employee.z_index = 2
	employees.append(employee)  # Добавляем в массив
	print("  [ШАГ 5] Сотрудник создан, employees.size()=", employees.size())

	# Сохраняем рабочее место
	employee.set_meta("workplace_pos", workplace_pos)

	# ШАГ 6: Привязка к окну
	employee_windows[window] = employee
	print("  [ШАГ 6] Сотрудник привязан к окну, employee_windows.size()=", employee_windows.size())

	# Находим индекс сотрудника для траектории
	var emp_index = employees.size() - 1
	
	# Отправляем к рабочему месту через промежуточную точку
	if emp_index < employee_spawn_paths.size():
		var via_point = employee_spawn_paths[emp_index]
		print("  [ШАГ 5] Траектория: дверь → ", via_point, " → ", workplace_pos)

		employee.go_to(via_point)
		employee.on_reached_callback = func():
			# Дошли до промежуточной точки → идём к столу
			employee.go_to(workplace_pos)
			employee.on_reached_callback = func():
				print("  [ШАГ 5] Сотрудник прибыл на рабочее место")
				# Новое окно — проверяем очередь
				_call_from_queue()
	else:
		# Если нет траектории — идём напрямую
		employee.go_to(workplace_pos)
		employee.on_reached_callback = func():
			print("  [ШАГ 5] Сотрудник прибыл на рабочее место")
			# Новое окно — проверяем очередь
			_call_from_queue()

func _on_purchase_quality(slot: int) -> void:
	if purchase_quality(slot):
		print("  Качество: ячейка %d куплена!" % (slot + 1))
		_update_progress_panel()
		_update_ui()
		_save_progress()
	else:
		print("  Недостаточно XP!")

func _on_purchase_upgrade(slot: int) -> void:
	if purchase_upgrade(slot):
		print("  Обновление: ячейка %d куплена!" % (slot + 1))
		_update_progress_panel()
		_update_ui()
		_save_progress()
	else:
		print("  Недостаточно XP!")

func _on_upgrade_system(level: int) -> void:
	if upgrade_system(level):
		print("  Система улучшена!")
		_update_progress_panel()
		_update_ui()
	else:
		print("  Недостаточно XP для улучшения!")

func _on_upgrade_queue(upgrade_index: int) -> void:
	if upgrade_index < 0 or upgrade_index >= queue_upgrades.size():
		return

	var upgrade = queue_upgrades[upgrade_index]
	if upgrade.purchased:
		return

	# Проверка: нельзя купить меньшее количество стульев
	if upgrade.chairs <= max_queue_size:
		print("  Нельзя купить меньшее количество стульев! Текущее: %d, пытаемся купить: %d" % [max_queue_size, upgrade.chairs])
		return

	if xp >= upgrade.cost:
		xp -= upgrade.cost
		max_queue_size = upgrade.chairs
		upgrade.purchased = true
		print("  Очередь улучшена до %d мест!" % max_queue_size)
		_update_waiting_chairs()
		_update_progress_panel()
		_update_ui()
		_save_progress()
	else:
		print("  Недостаточно XP!")

func _on_activate_boost(boost_id: String) -> void:
	if activate_boost(boost_id):
		print("  Буст активирован!")
		_update_progress_panel()
		_update_ui()
		_save_progress()
	else:
		print("  Недостаточно XP!")

func _create_waiting_chairs(count: int) -> void:
	for i in range(count):
		var chair = Sprite2D.new()
		chair.name = "WaitingChair_%d" % i
		chair.texture = CHAIR_WAITING_TEX
		chair.scale = Vector2(0.25, 0.25)
		var px = WAITING_START.x + (i % 5) * WAITING_CHAIR_SPACING
		var py = WAITING_START.y + floori(i / 5.0) * 100
		chair.position = Vector2(px, py)
		chair.z_index = 2
		add_child(chair)
		waiting_chairs.append(chair)

func _update_waiting_chairs() -> void:
	# Пересоздаём стулья при изменении max_queue_size
	# Сначала удаляем старые
	for chair in waiting_chairs:
		if is_instance_valid(chair):
			chair.queue_free()
	waiting_chairs.clear()
	
	# Создаём новые (всегда 10)
	_create_waiting_chairs(10)
	print("  Стулья обновлены: %d мест" % max_queue_size)
	_update_waiting_chairs_visibility()

func _update_waiting_chairs_visibility() -> void:
	# Показываем только первые max_queue_size стульев
	for i in range(waiting_chairs.size()):
		if is_instance_valid(waiting_chairs[i]):
			if i < max_queue_size:
				waiting_chairs[i].visible = true
			else:
				waiting_chairs[i].visible = false

func _spawn_visitor() -> void:
	var visitor = VISITOR_SCENE.instantiate()

	# Случайный уникальный посетитель (1-40) — ДО add_child!
	visitor.visitor_index = randi_range(1, 40)

	add_child(visitor)
	visitor.position = DOOR_BOTTOM

	# Репутация влияет на типы: выше репутация → больше сложных (больше XP)
	var simple_chance = lerp(0.65, 0.30, reputation / 100.0)
	var medium_chance = lerp(0.30, 0.35, reputation / 100.0)
	# Complex = остаток

	var roll = randf()
	if roll < simple_chance:
		visitor.visitor_type = Visitor.VisitorType.SIMPLE
	elif roll < simple_chance + medium_chance:
		visitor.visitor_type = Visitor.VisitorType.MEDIUM
	else:
		visitor.visitor_type = Visitor.VisitorType.COMPLEX

	# Терпение зависит от уровня системы: 150 → 135 → 120 → 105 → 90
	# + бонус от декора: +15 сек за каждый элемент
	var base_patience = 150.0 - (system_level * 15.0)
	var decor_patience = office_decor_system.get_patience_bonus_seconds()
	visitor.max_patience = base_patience + decor_patience

	# Движение к инфомату: сначала Y, потом X
	visitor.go_to_infomat(INFOMAT_POS)
	visitor.on_reached_callback = func(): _at_infomat(visitor)

func _at_infomat(visitor: Visitor) -> void:
	if game_paused:
		return

	# Посетитель у инфомата - "взаимодействует" 1-2 секунды
	print("Посетитель взаимодействует с инфоматом...")
	await get_tree().create_timer(randf_range(1.0, 2.0)).timeout

	# Проверяем есть ли СВОБОДНЫЕ окна
	var window_free = _get_free_window() != null

	if window_free:
		# Есть свободное окно - к окну
		_go_to_window(visitor)
	elif queue.size() < max_queue_size:
		# Есть место в очереди
		_go_to_queue(visitor)
	else:
		# Нет мест - уходим
		_go_home(visitor)

func _go_to_window(visitor: Visitor) -> void:
	# 🔵 СИНИЙ маршрут: инфомат/зал → окно
	# Путь: вправо → вверх → вправо → вниз (обход рабочих мест)
	var window = _get_free_window()
	if not window:
		# Все окна заняты - в очередь
		_go_to_queue(visitor)
		return

	# Помечаем окно как занятое
	window.is_serving = true
	print("[ШАГ 7] Окно #", windows.find(window) + 1, " помечено как занятое (посетитель от инфомата)")

	# Посетитель идёт к этому окну по синему маршруту
	visitor.z_index = 6  # При движении: больше стула (поверх)
	visitor.go_to_window(window.position + Vector2(0, 30))
	visitor.on_reached_callback = func(): 
		visitor.z_index = 4  # При посадке: меньше стула (сидит на стуле)
		_start_service(visitor, window)
	print("[ШАГ 7] Посетитель идёт к окну #", windows.find(window) + 1, " на позиции ", window.position)

# ШАГ 7: Поиск свободного окна (сотрудник НА МЕСТЕ!)
func _get_free_window() -> ServiceWindow:
	for window in windows:
		# Окно свободно ТОЛЬКО если:
		# 1. Не обслуживается
		# 2. Сотрудник на месте (НЕ у шкафа)
		if not window.is_serving:
			if window in employee_windows:
				var employee = employee_windows[window]
				# Проверяем что сотрудник НЕ у шкафа
				if employee not in employees_busy or not employees_busy[employee]:
					return window
	return null

func _go_to_queue(visitor: Visitor) -> void:
	# Собираем какие стулья ЗАНЯТЫ (из очереди + те кто ещё идёт к стулу)
	var occupied_chairs = []
	
	# 1. Проверяем очередь
	for v in queue:
		if is_instance_valid(v) and v.has_meta("chair_idx"):
			occupied_chairs.append(v.get_meta("chair_idx"))
	
	# 2. Проверяем ВСЕХ посетителей в сцене (даже тех кто ещё не в очереди)
	var all_visitors = get_tree().get_nodes_in_group("visitors")
	for v in all_visitors:
		if is_instance_valid(v) and v != visitor and v.has_meta("chair_idx"):
			var idx = v.get_meta("chair_idx")
			if not occupied_chairs.has(idx):
				occupied_chairs.append(idx)
	
	print("  Заняты стулья: ", occupied_chairs, " из ", max_queue_size)
	
	# Проверяем есть ли место
	if occupied_chairs.size() >= max_queue_size:
		# Нет мест - уходим
		print("  Нет мест в очереди - посетитель уходит")
		_go_home(visitor)
		return

	# Находим первый СВОБОДНЫЙ стул
	for i in range(max_queue_size):
		if not occupied_chairs.has(i):
			if i < waiting_chairs.size() and is_instance_valid(waiting_chairs[i]):
				# БЛОКИРУЕМ стул СРАЗУ (пока посетитель ещё не дошёл)
				print("  Посетитель садится на стул %d" % i)
				visitor.set_meta("chair_idx", i)

				var pos = Vector2(
					WAITING_START.x + (i % 5) * WAITING_CHAIR_SPACING,
					WAITING_START.y + floori(i / 5.0) * 100 - 15
				)
				# 🟡 ЖЁЛТЫЙ маршрут: инфомат → зал ожидания
				visitor.z_index = 3  # При движении: больше стула (поверх)
				visitor.go_to_waiting(pos, i)
				visitor.on_reached_callback = func(): 
					visitor.z_index = 1  # При посадке: меньше стула (стул поверх)
					_in_queue(visitor)
				return
			else:
				_go_home(visitor)
				return
	
	_go_home(visitor)

func _go_home(visitor: Visitor) -> void:
	# 🟣 ФИОЛЕТОВЫЙ маршрут: инфомат → дверь (уход)
	visitor.go_to_door(DOOR_BOTTOM)
	visitor.on_reached_callback = func(): _remove_visitor(visitor)

func _in_queue(visitor: Visitor) -> void:
	visitor.state = "waiting"
	# Добавляем в очередь (chair_idx уже установлен)
	queue.append(visitor)
	print("  Посетитель в очереди (всего: %d/%d, стул: %s)" % [
		queue.size(), max_queue_size,
		visitor.get_meta("chair_idx") if visitor.has_meta("chair_idx") else "?"
	])

	# Callback когда посетитель устал ждать
	visitor.on_impatient_callback = func(): _visitor_left_impatient(visitor)

	# НЕ вызываем _call_from_queue здесь! Он вызывается после обслуживания!

func _visitor_left_impatient(visitor: Visitor) -> void:
	if not is_instance_valid(visitor):
		return

	# Удаляем из очереди
	if queue.has(visitor):
		queue.erase(visitor)
		print("  Посетитель ушёл из очереди (нетерпеливость)!")

		# Освобождаем стул
		if visitor.has_meta("chair_idx"):
			var chair_idx = visitor.get_meta("chair_idx")
			visitor.remove_meta("chair_idx")
			print("  Освободился стул %d" % chair_idx)

		# Репутация падает
		left_count += 1
		reputation = clamp(reputation - 5, 0, 100)
		_update_ui()
		print("  Репутация: %d (ушедших: %d)" % [reputation, left_count])

	# Посетитель уходит к двери
	visitor.go_to_door(DOOR_BOTTOM)
	visitor.on_reached_callback = func(): _remove_visitor(visitor)

func _start_service(visitor: Visitor, window: ServiceWindow) -> void:
	visitor.state = "at_window"

	# Время обслуживания зависит от уровня системы
	var service_time = get_service_time()
	print("Обслуживание: %.1f сек (уровень системы: %d)" % [service_time, system_level])
	print("[ШАГ 7] Обслуживание у окна #", windows.find(window) + 1)

	if window:
		window.start_service(service_time)
	else:
		print("[_start_service] ОШИБКА: окно НЕ найдено!")
		return  # Не продолжаем если окно не найдено

	# Ждём время обслуживания с учётом паузы
	print("[_start_service] НАЧАЛО таймера на %.1f сек" % service_time)

	# Обновляем прогресс-бар во время обслуживания
	var elapsed_time = 0.0
	var update_interval = 0.1  # Обновляем каждые 0.1 сек

	while elapsed_time < service_time:
		await get_tree().create_timer(update_interval).timeout
		# Увеличиваем время только если не пауза
		if not game_paused:
			elapsed_time += update_interval
			# Обновляем прогресс в окне
			if window and window.is_serving:
				window.service_time_remaining -= update_interval
				window.service_progress.emit(window.service_time_remaining, window.service_time_total)

	print("[_start_service] ТАЙМЕР ЗАВЕРШЁН")
	
	# Проверяем что посетитель ещё здесь
	if not is_instance_valid(visitor):
		print("[_start_service] Посетитель исчез, прерываем обслуживание")
		return
	
	# Завершаем обслуживание в окне
	if window:
		window.is_serving = false
		window.service_completed.emit()
		print("[_start_service] service_completed отправлен")

	# Посетитель обслужен - считаем обслуженного
	served_count += 1
	var rep_bonus = office_decor_system.get_reputation_bonus()
	var rep_gain = 2 + rep_bonus  # 2 базовая + бонус от декора
	reputation = clamp(reputation + rep_gain, 0, 100)
	_update_ui()
	print("Обслужено: %d, Репутация: %d (+%d%s)" % [served_count, reputation, rep_gain,
		" (%d декор)" % rep_bonus if rep_bonus > 0 else ""])

	# Проверяем достижения
	_check_achievements()

	# ШАГ 8: СНАЧАЛА отправляем сотрудника к шкафу
	# Сохраняем тип посетителя в окне чтобы сотрудник получил XP
	window.set_meta("last_visitor_type", visitor.visitor_type)
	window.set_meta("last_visitor_index", visitor.visitor_index)
	_employee_go_to_cabinet(window)

	# ПОТОМ вызываем следующего из очереди (чтобы он мог занять ДРУГОЕ свободное окно)
	_call_from_queue()

	# Посетитель уходит
	# 🟣 ФИОЛЕТОВЫЙ маршрут: окно → дверь
	visitor.go_to_door(DOOR_BOTTOM)
	visitor.on_reached_callback = func(): _remove_visitor(visitor)

func _employee_go_to_cabinet(window: ServiceWindow) -> void:
	# Находим сотрудника который обслуживал у этого окна
	var employee = null
	if window in employee_windows:
		employee = employee_windows[window]
		print("[ШАГ 8] Сотрудник окна #", windows.find(window) + 1, " идёт к шкафу")
	else:
		print("[ШАГ 8] ОШИБКА: Сотрудник не найден для окна #", windows.find(window) + 1)
		return

	# Помечаем сотрудника как занятого
	employees_busy[employee] = true

	# Сохраняем тип посетителя в сотруднике для XP
	if window.has_meta("last_visitor_type"):
		employee.set_meta("last_visitor_type", window.get_meta("last_visitor_type"))
	if window.has_meta("last_visitor_index"):
		employee.set_meta("last_visitor_index", window.get_meta("last_visitor_index"))

	# Траектория к шкафу:
	# 1. Вверх на 20 пикселей
	# 2. Вправо (x) до уровня шкафа (x=1150)
	# 3. Вверх/вниз (y) до шкафа (y=250)
	employee.move_pattern = "xy"

	var workplace_pos = employee.get_meta("workplace_pos")
	if not workplace_pos:
		# Запасной вариант для первого сотрудника
		workplace_pos = EMPLOYEE_CHAIR
		employee.set_meta("workplace_pos", workplace_pos)
		print("  [ШАГ 8] workplace_pos не найден, используем запасной: ", workplace_pos)

	# Точка 1: Вверх на 20 пикселей от стола
	var point1 = Vector2(workplace_pos.x, workplace_pos.y - 20)
	# Точка 2: Вправо до уровня шкафа (на той же высоте)
	var point2 = Vector2(1150, workplace_pos.y - 20)

	# Сохраняем точку возврата (point2)
	employee.set_meta("cabinet_via_point", point2)

	# Сначала идём вверх на 20 пикселей
	employee.go_to(point1)
	employee.on_reached_callback = func():
		# Теперь идём вправо до уровня шкафа
		employee.go_to(point2)
		employee.on_reached_callback = func():
			# Теперь идём к шкафу
			employee.go_to(CABINET_POS)
			employee.on_reached_callback = func(): _employee_at_cabinet(employee)

func _employee_at_cabinet(employee: Node) -> void:
	# Сотрудник у шкафа - кладёт документы (2 секунды)
	print("Сотрудник кладёт документы в шкаф (2 сек)...")

	# Начисляем XP по типу посетителя
	var visitor_type = Visitor.VisitorType.SIMPLE  # По умолчанию
	var visitor_index = 1  # По умолчанию
	if employee.has_meta("last_visitor_type"):
		visitor_type = employee.get_meta("last_visitor_type")
	if employee.has_meta("last_visitor_index"):
		visitor_index = employee.get_meta("last_visitor_index")

	var base_xp = Visitor.XP_REWARDS.get(visitor_type, 3)
	var is_vip = visitor_index in Visitor.VIP_VISITORS
	var xp_reward = int(base_xp * Visitor.VIP_XP_MULTIPLIER) if is_vip else base_xp
	var decor_bonus = office_decor_system.get_xp_bonus_multiplier()
	var total_xp = int(xp_reward * decor_bonus)
	xp += total_xp
	_update_ui()
	var vip_tag = " ⭐VIP" if is_vip else ""
	print("XP: %d (+%d за тип %s%s%s)" % [xp, total_xp, ["Простой", "Средний", "Сложный"][visitor_type], vip_tag,
		" (декор x%.0f%%)" % (decor_bonus * 100) if decor_bonus > 1.0 else ""])

	await get_tree().create_timer(2.0).timeout

	# Сотрудник возвращается к столу ПО ОБРАТНОЙ ТРАЕКТОРИИ:
	# 1. Вверх/вниз (y) до уровня point2 (Y стола - 20)
	# 2. Влево (x) до X стола
	# 3. Вниз на 20 пикселей к столу
	print("Сотрудник возвращается к окну...")

	var workplace_pos = employee.get_meta("workplace_pos")
	# Точка 2: Вправо до уровня шкафа (на той же высоте) - уже сохранена
	var point2 = employee.get_meta("cabinet_via_point")
	# Точка 1: Вверх на 20 пикселей от стола
	var point1 = Vector2(workplace_pos.x, workplace_pos.y - 20)

	# Сначала идём вверх/вниз до уровня point2
	employee.move_pattern = "xy"
	employee.go_to(point2)
	employee.on_reached_callback = func():
		# Теперь идём влево до X стола
		employee.go_to(point1)
		employee.on_reached_callback = func():
			# Теперь идём вниз к рабочему месту
			employee.go_to(workplace_pos)
			employee.on_reached_callback = func():
				# Сотрудник вернулся - снимаем флаг занятости
				employees_busy[employee] = false
				print("Сотрудник на месте, окно свободно")
				# Вызываем следующего из очереди!
				_call_from_queue()

func _employee_back_at_desk() -> void:
	# Заглушка - теперь _call_from_queue() вызывается при возвращении
	pass

func _call_from_queue() -> void:
	if queue.size() > 0:
		var next_visitor = queue.pop_front()
		if is_instance_valid(next_visitor):
			# СРАЗУ удаляем meta чтобы другие могли сесть на это место
			if next_visitor.has_meta("chair_idx"):
				var chair_idx = next_visitor.get_meta("chair_idx")
				next_visitor.remove_meta("chair_idx")
				print("  Освободился стул %d" % chair_idx)

			# ШАГ 7: Находим ЛЮБОЕ свободное окно (не обязательно то что освободилось!)
			var window = _get_free_window()
			if window:
				# СРАЗУ помечаем окно как занятое!
				window.is_serving = true
				print("[ШАГ 7] Окно #", windows.find(window) + 1, " помечено как занятый (посетитель из очереди)")
				var visitor_chair_pos = window.position + Vector2(0, 30)
				next_visitor.z_index = 6  # При движении: больше стула (поверх)
				next_visitor.go_to_window(visitor_chair_pos)
				next_visitor.on_reached_callback = func(): 
					next_visitor.z_index = 4  # При посадке: меньше стула (сидит на стуле)
					_start_service(next_visitor, window)
				print("[ШАГ 7] Посетитель из очереди идёт к окну #", windows.find(window) + 1)
			else:
				# Нет свободных окон - возвращаем в очередь
				queue.insert(0, next_visitor)
				print("  Нет свободных окон, возвращаем в очередь")

func _remove_visitor(visitor: Visitor) -> void:
	await get_tree().create_timer(1.0).timeout
	if is_instance_valid(visitor):
		# Удаляем meta чтобы стул освободился
		if visitor.has_meta("chair_idx"):
			visitor.remove_meta("chair_idx")
		visitor.queue_free()

func _update_ui() -> void:
	var ui = get_node_or_null("UI")
	if not ui:
		return

	var xp_label = ui.get_node_or_null("XPLabel")
	var served_label = ui.get_node_or_null("ServedLabel")
	var service_time_label = ui.get_node_or_null("ServiceTimeLabel")
	var rep_label = ui.get_node_or_null("ReputationLabel")
	var boost_timer_label = ui.get_node_or_null("BoostTimerLabel")

	if xp_label:
		xp_label.text = "XP: %d" % xp
	if served_label:
		served_label.text = "Обслужено: %d" % served_count
	if rep_label:
		var stars = ""
		if reputation >= 90: stars = "★★★★★"
		elif reputation >= 70: stars = "★★★★☆"
		elif reputation >= 50: stars = "★★★☆☆"
		elif reputation >= 30: stars = "★★☆☆☆"
		else: stars = "★☆☆☆☆"
		rep_label.text = "Репутация: %d %s" % [reputation, stars]
	if service_time_label:
		var service_time = get_service_time()
		service_time_label.text = "Время: %.0f сек" % service_time
	if boost_timer_label:
		# Показываем активные бусты
		var boost_texts = []
		for boost_id in active_boosts.keys():
			if boost_id in boost_definitions:
				boost_texts.append("%s: %.0f" % [
					boost_definitions[boost_id].name,
					active_boosts[boost_id]
				])
		boost_timer_label.text = " | ".join(boost_texts) if boost_texts else ""
	
	# Обновляем панель прогресса если открыта
	var progress_panel = ui.get_node_or_null("ProgressPanel")
	if progress_panel and progress_panel.visible:
		_update_progress_panel()

func _update_boost_ui() -> void:
	# Обновляем UI бустов в реальном времени
	var ui = get_node_or_null("UI")
	if not ui:
		return

	# 1. Обновляем BoostTimerLabel в главном HUD
	var boost_timer_label = ui.get_node_or_null("BoostTimerLabel")
	if boost_timer_label:
		var boost_texts = []
		for boost_id in active_boosts.keys():
			if boost_id in boost_definitions:
				boost_texts.append("%s: %.0f" % [
					boost_definitions[boost_id].name,
					active_boosts[boost_id]
				])
		boost_timer_label.text = " | ".join(boost_texts) if boost_texts else ""

	# 2. Обновляем вкладку "Бусты" если панель прогресса открыта
	var progress_panel = ui.get_node_or_null("ProgressPanel")
	if not progress_panel or not progress_panel.visible:
		return

	# Полная перерисовка вкладки (чтобы обновить панель сверху)
	_fill_boosts_tab(ui)

func get_service_time() -> float:
	# Базовое время от уровня системы
	var base_time = system_levels[system_level].time
	
	# Улучшения качества (-1 сек за ячейку)
	for i in range(system_quality.size()):
		if system_quality[i]:
			base_time -= 1.0
	
	# Улучшения обновления (-5 сек за ячейку)
	for i in range(system_upgrade.size()):
		if system_upgrade[i]:
			base_time -= 5.0
	
	# Бусты (множитель)
	var boost_multiplier = 1.0
	for boost_id in active_boosts.keys():
		if boost_id in boost_definitions:
			boost_multiplier *= boost_definitions[boost_id].multiplier
	
	base_time *= boost_multiplier
	
	return max(base_time, 5.0)  # Минимум 5 секунд

func activate_boost(boost_id: String) -> bool:
	if boost_id not in boost_definitions:
		return false

	var boost = boost_definitions[boost_id]
	if xp < boost.cost:
		return false

	xp -= boost.cost

	# Добавляем или обновляем буст
	active_boosts[boost_id] = boost.duration
	print("  Буст активирован: %s на %.0f сек" % [boost.name, boost.duration])

	return true

# === Улучшения системы ===

func can_purchase_quality(slot: int) -> bool:
	if slot < 0 or slot >= system_quality.size():
		return false
	if system_quality[slot]:
		return false  # Уже куплено
	# Цена увеличивается с уровнем системы
	var cost_mult = pow(2, system_level)
	return xp >= system_quality_costs[slot] * cost_mult

func purchase_quality(slot: int) -> bool:
	if not can_purchase_quality(slot):
		return false
	
	var cost_mult = pow(2, system_level)
	var cost = system_quality_costs[slot] * cost_mult
	
	xp -= cost
	system_quality[slot] = true
	print("  Качество улучшено: ячейка %d (-1 сек)" % (slot + 1))
	return true

func can_purchase_upgrade(slot: int) -> bool:
	if slot < 0 or slot >= system_upgrade.size():
		return false
	if system_upgrade[slot]:
		return false  # Уже куплено
	# Цена увеличивается с уровнем системы
	var cost_mult = pow(2, system_level)
	return xp >= system_upgrade_costs[slot] * cost_mult

func purchase_upgrade(slot: int) -> bool:
	if not can_purchase_upgrade(slot):
		return false
	
	var cost_mult = pow(2, system_level)
	var cost = system_upgrade_costs[slot] * cost_mult
	
	xp -= cost
	system_upgrade[slot] = true
	print("  Обновление улучшено: ячейка %d (-5 сек)" % (slot + 1))
	return true

func _process(delta: float) -> void:
	# Обновляем таймеры бустов (даже на паузе)
	for boost_id in active_boosts.keys():
		active_boosts[boost_id] -= delta
		if active_boosts[boost_id] <= 0:
			active_boosts.erase(boost_id)
			print("  Буст закончился: %s" % boost_definitions[boost_id].name)

	# Обновляем UI бустов каждые 0.5 сек (если панель прогресса открыта)
	boost_ui_timer += delta
	if boost_ui_timer >= 0.5:
		boost_ui_timer = 0.0
		_update_boost_ui()

	if game_paused:
		return

	# Случайные события
	var event = random_events_system._process(delta, reputation, game_paused)
	if event.size() > 0:
		_trigger_event(event)

	visitor_spawn_timer -= delta
	if visitor_spawn_timer <= 0:
		_spawn_visitor()
		# Репутация влияет на частоту: выше репутация → чаще
		var spawn_min = lerp(5.0, 1.5, reputation / 100.0)
		var spawn_max = lerp(25.0, 8.0, reputation / 100.0)
		visitor_spawn_timer = randf_range(spawn_min, spawn_max)

func can_upgrade_system() -> bool:
	# Можно ли улучшить систему
	if system_level >= system_levels.size() - 1:
		return false  # Максимальный уровень
	var next_level = system_level + 1
	return xp >= system_levels[next_level].cost

func upgrade_system(target_level: int = -1) -> bool:
	# Улучшить систему до target_level (или следующего если -1)
	if target_level < 0:
		target_level = system_level + 1

	if target_level <= system_level:
		return false  # Нельзя купить уровень ниже или равный текущему

	if target_level >= system_levels.size():
		return false  # Несуществующий уровень

	if xp < system_levels[target_level].cost:
		return false  # Недостаточно XP

	xp -= system_levels[target_level].cost
	system_level = target_level

	# Сбрасываем улучшения при покупке новой ступени
	system_quality = [false, false, false, false, false]
	system_upgrade = [false, false, false, false, false]

	print("Система улучшена до уровня %d: %s" % [system_level, system_levels[system_level].name])
	print("  Улучшения сброшены (качество и обновление)")
	_save_progress()
	return true

# === Сохранение / Загрузка ===

func _load_progress() -> void:
	if SaveSystem.load_game():
		# Применяем загруженные данные
		xp = SaveSystem.data.xp
		served_count = SaveSystem.data.served_count
		reputation = SaveSystem.data.reputation
		left_count = SaveSystem.data.left_count
		system_level = SaveSystem.data.system_level
		# Конвертируем Array -> Array[bool]
		var sq = SaveSystem.data.system_quality
		for i in range(system_quality.size()):
			system_quality[i] = sq[i]
		var su = SaveSystem.data.system_upgrade
		for i in range(system_upgrade.size()):
			system_upgrade[i] = su[i]
		max_queue_size = SaveSystem.data.max_queue_size
		queue_upgrades = SaveSystem.data.queue_upgrades
		employee_cost = SaveSystem.data.employee_cost

		# Загружаем достижения
		if SaveSystem.data.has("achievements"):
			achievements_system.load_unlocked(SaveSystem.data.achievements)

		# Загружаем декор
		if SaveSystem.data.has("office_decor"):
			office_decor_system.load_decor_data(SaveSystem.data.office_decor)

		# Обновляем UI после загрузки
		_update_ui()
		_update_waiting_chairs()

		# Доспавниваем сотрудников если их больше 1
		var saved_count = SaveSystem.data.employee_count
		for i in range(1, saved_count):
			if i < employee_positions.size():
				_activate_workplace(i)
				_spawn_employee(employee_positions[i], windows[i])

		print("[Save] Прогресс загружен")
	else:
		print("[Save] Нет сохранения, начинаем заново")

func _save_progress() -> void:
	SaveSystem.data.xp = xp
	SaveSystem.data.served_count = served_count
	SaveSystem.data.reputation = reputation
	SaveSystem.data.left_count = left_count
	SaveSystem.data.system_level = system_level
	SaveSystem.data.system_quality = system_quality
	SaveSystem.data.system_upgrade = system_upgrade
	SaveSystem.data.max_queue_size = max_queue_size
	SaveSystem.data.queue_upgrades = queue_upgrades
	SaveSystem.data.employee_count = employees.size()
	SaveSystem.data.employee_cost = employee_cost
	SaveSystem.data.achievements = achievements_system.get_unlocked()
	SaveSystem.data.office_decor = office_decor_system.get_decor_data_for_save()
	SaveSystem.save_game()

func _check_achievements() -> void:
	var new_achs = achievements_system.check_achievements(
		served_count, employees.size(), system_level,
		reputation, left_count, max_queue_size, xp
	)

	for ach_id in new_achs:
		var data = achievements_system.get_achievement_data(ach_id)
		print("🏆 ДОСТИЖЕНИЕ: %s — %s (+%d XP)" % [data.name, data.desc, data.xp_reward])
		xp += data.xp_reward

		# Показываем уведомление
		if active_event_notification:
			active_event_notification.text = "🏆 %s — %s (+%d XP)" % [data.name, data.desc, data.xp_reward]
			active_event_notification.add_theme_color_override("font_color", Color(1, 0.85, 0, 1))
			active_event_notification.visible = true
			await get_tree().create_timer(8.0).timeout
			if active_event_notification:
				active_event_notification.visible = false

	if new_achs.size() > 0:
		_update_ui()
		_save_progress()

func _trigger_event(event: Dictionary) -> void:
	print("[СОБЫТИЕ] %s (%s)" % [event.name, event.type])

	# Применяем эффекты
	if event.has("xp") and event.xp > 0:
		xp += event.xp
	if event.has("rep") and event.rep != 0:
		reputation = clamp(reputation + event.rep, 0, 100)

	# Штраф к времени обслуживания
	if event.has("service_penalty") and event.service_penalty > 0:
		print("  Штраф к обслуживанию: +%d сек" % event.service_penalty)

	_update_ui()

	# Показываем уведомление
	if active_event_notification:
		var color = Color(1, 1, 0.5, 1)  # Жёлтый по умолчанию
		if event.type == "good":
			color = Color(0.3, 1, 0.3, 1)  # Зелёный
		elif event.type == "bad":
			color = Color(1, 0.4, 0.4, 1)  # Красный

		active_event_notification.text = "📢 %s — %s" % [event.name, event.text]
		active_event_notification.add_theme_color_override("font_color", color)
		active_event_notification.visible = true

		# Скрываем через 10 секунд
		await get_tree().create_timer(10.0).timeout
		if active_event_notification:
			active_event_notification.visible = false

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		_save_progress()
