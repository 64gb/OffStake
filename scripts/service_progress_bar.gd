class_name ServiceProgressBar
extends ProgressBar

## Полоска прогресса обслуживания
## Серая полоска заполняется зелёным по мере обслуживания

@export var service_window: ServiceWindow = null

func _ready() -> void:
	# Настраиваем прогресс бар
	min_value = 0.0
	max_value = 100.0
	value = 0.0  # Начинаем с пустого (будет заполняться)
	show_percentage = false
	
	# Скрываем прогресс бар изначально
	visible = false
	
	print("[ServiceProgressBar] _ready() вызван")
	
	# Создаём стили для прогресс бара
	var bg_style = StyleBoxFlat.new()
	bg_style.bg_color = Color(0.3, 0.3, 0.3, 1)  # Серый фон
	
	var fill_style = StyleBoxFlat.new()
	fill_style.bg_color = Color(0.2, 0.8, 0.2, 1)  # Зелёное заполнение
	
	# Применяем стили
	add_theme_stylebox_override("background", bg_style)
	add_theme_stylebox_override("fill", fill_style)
	
	# Находим окно в родителе
	var parent = get_parent()
	print("[ServiceProgressBar] parent = ", parent)
	
	if parent is ServiceWindow:
		service_window = parent
		service_window.service_started.connect(_on_service_started)
		service_window.service_progress.connect(_on_service_progress)
		service_window.service_completed.connect(_on_service_completed)
		print("[ServiceProgressBar] подключён к сигналам ServiceWindow")
	else:
		print("[ServiceProgressBar] ОШИБКА: parent не ServiceWindow!")

func _on_service_started() -> void:
	# Показываем прогресс бар когда началось обслуживание
	print("[ServiceProgressBar] _on_service_started()")
	visible = true
	value = 0.0  # Начинаем с нуля (будет заполняться)

func _on_service_progress(current: float, total: float) -> void:
	if total > 0 and visible:
		# Рассчитываем прошедшее время и заполняем бар
		var elapsed = total - current
		value = (elapsed / total) * 100.0
		# print("[ServiceProgressBar] progress: ", value, "%")

func _on_service_completed() -> void:
	# Скрываем прогресс бар когда обслуживание закончилось
	print("[ServiceProgressBar] _on_service_completed()")
	visible = false
	value = 0.0  # Сбрасываем для следующего раза
