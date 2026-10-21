## Система сохранений
## Сохраняет прогресс в user://savegame.json

const SAVE_PATH = "user://savegame.json"

# Все сохраняемые данные
static var data: Dictionary = {
	"xp": 100,
	"served_count": 0,
	"reputation": 50,
	"left_count": 0,
	"system_level": 0,
	"system_quality": [false, false, false, false, false],
	"system_upgrade": [false, false, false, false, false],
	"max_queue_size": 2,
	"queue_upgrades": [
		{"chairs": 2, "cost": 0, "purchased": true},
		{"chairs": 3, "cost": 100, "purchased": false},
		{"chairs": 5, "cost": 400, "purchased": false},
		{"chairs": 7, "cost": 800, "purchased": false},
		{"chairs": 10, "cost": 1500, "purchased": false}
	],
	"employee_count": 1,
	"employee_cost": 150,
	"achievements": [],
	"office_decor": {},
	"boost_purchase_counts": {},
}


## Сохранить данные в файл
static func save_game() -> bool:
	var file = FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	if not file:
		print("[SaveSystem] ОШИБКА: Не удалось открыть файл для записи!")
		return false

	var json_string = JSON.stringify(data, "\t")
	file.store_string(json_string)
	file.close()
	print("[SaveSystem] Игра сохранена")
	return true


## Загрузить данные из файла
static func load_game() -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		print("[SaveSystem] Сохранение не найдено")
		return false

	var file = FileAccess.open(SAVE_PATH, FileAccess.READ)
	if not file:
		print("[SaveSystem] ОШИБКА: Не удалось открыть файл для чтения!")
		return false

	var json = JSON.new()
	var parse_result = json.parse(file.get_as_text())
	file.close()

	if parse_result != OK:
		print("[SaveSystem] ОШИБКА парсинга JSON: ", json.get_error_message())
		return false

	var loaded_data = json.data

	# Валидируем и загружаем поля
	if loaded_data.has("xp"):
		data.xp = loaded_data.xp
	if loaded_data.has("served_count"):
		data.served_count = loaded_data.served_count
	if loaded_data.has("system_level"):
		data.system_level = loaded_data.system_level
	if loaded_data.has("system_quality"):
		data.system_quality = loaded_data.system_quality
	if loaded_data.has("system_upgrade"):
		data.system_upgrade = loaded_data.system_upgrade
	if loaded_data.has("max_queue_size"):
		data.max_queue_size = loaded_data.max_queue_size
	if loaded_data.has("queue_upgrades"):
		data.queue_upgrades = loaded_data.queue_upgrades
	if loaded_data.has("employee_count"):
		data.employee_count = loaded_data.employee_count
	if loaded_data.has("employee_cost"):
		data.employee_cost = loaded_data.employee_cost
	if loaded_data.has("office_decor"):
		data.office_decor = loaded_data.office_decor
	if loaded_data.has("boost_purchase_counts"):
		data.boost_purchase_counts = loaded_data.boost_purchase_counts

	print("[SaveSystem] Игра загружена: XP=", data.xp, ", уровень=", data.system_level)
	return true


## Проверить существует ли сохранение
static func has_save() -> bool:
	return FileAccess.file_exists(SAVE_PATH)


## Удалить сохранение
static func delete_save() -> bool:
	if has_save():
		var err = DirAccess.remove_absolute(SAVE_PATH)
		if err == OK:
			print("[SaveSystem] Сохранение удалено")
			return true
		else:
			print("[SaveSystem] ОШИБКА при удалении: ", err)
			return false
	return false


## Сбросить данные на начальные
static func reset_data() -> void:
	data = {
		"xp": 100,
		"served_count": 0,
		"reputation": 50,
		"left_count": 0,
		"system_level": 0,
		"system_quality": [false, false, false, false, false],
		"system_upgrade": [false, false, false, false, false],
		"max_queue_size": 2,
		"queue_upgrades": [
			{"chairs": 2, "cost": 0, "purchased": true},
			{"chairs": 3, "cost": 100, "purchased": false},
			{"chairs": 5, "cost": 400, "purchased": false},
			{"chairs": 7, "cost": 800, "purchased": false},
			{"chairs": 10, "cost": 1500, "purchased": false}
		],
		"employee_count": 1,
		"employee_cost": 150,
		"achievements": [],
		"office_decor": {},
		"boost_purchase_counts": {},
	}
