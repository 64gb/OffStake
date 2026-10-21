class_name ServiceWindow
extends Node2D

## Окно приёма - рабочее место сотрудника

signal service_started()
signal service_completed()
signal service_progress(current: float, total: float)

@export var window_id: int = 0
@export var desk_position: Vector2 = Vector2.ZERO
@export var chair_position: Vector2 = Vector2.ZERO
@export var cabinet_position: Vector2 = Vector2.ZERO

var has_employee: bool = false  # Сотрудник сидит за окном
var is_serving: bool = false  # Идёт обслуживание
var service_time_remaining: float = 0.0
var service_time_total: float = 0.0

func _ready() -> void:
	# Проверяем что ServiceProgress существует
	var progress_bar = get_node_or_null("ServiceProgress")
	if progress_bar:
		print("[ServiceWindow] ServiceProgress найден: ", progress_bar)
	else:
		print("[ServiceWindow] ОШИБКА: ServiceProgress НЕ найден!")

func set_positions(desk_pos: Vector2, chair_pos: Vector2, cabinet_pos: Vector2) -> void:
	desk_position = desk_pos
	chair_position = chair_pos
	cabinet_position = cabinet_pos

func is_free() -> bool:
	# Окно свободно если нет обслуживания
	return not is_serving

func start_service(duration: float) -> void:
	is_serving = true
	service_time_total = duration
	service_time_remaining = duration
	print("[ServiceWindow] start_service() duration=", duration)
	service_started.emit()
	service_progress.emit(service_time_remaining, service_time_total)

func update_service(delta: float) -> float:
	if not is_serving:
		return 0.0
	service_time_remaining -= delta
	if service_time_remaining <= 0:
		service_time_remaining = 0
		# НЕ отправляем service_completed здесь - это делает _start_service() через таймер
	service_progress.emit(service_time_remaining, service_time_total)
	return service_time_remaining

func get_cabinet_position() -> Vector2:
	return cabinet_position

func get_desk_position() -> Vector2:
	return desk_position

func get_chair_position() -> Vector2:
	return chair_position
