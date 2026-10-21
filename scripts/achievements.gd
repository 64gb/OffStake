## Система достижений
## Отслеживает прогресс и выдаёт награды

const Achievement = {
	"first_visitor": {
		"name": "Первый посетитель",
		"desc": "Обслужить первого посетителя",
		"xp_reward": 20,
	},
	"conveyor": {
		"name": "Конвейер",
		"desc": "Обслужить 50 посетителей",
		"xp_reward": 100,
		"require_served": 50,
	},
	"veteran": {
		"name": "Ветеран труда",
		"desc": "Обслужить 200 посетителей",
		"xp_reward": 500,
		"require_served": 200,
	},
	"businessman": {
		"name": "Бизнесмен",
		"desc": "Нанять 5 сотрудников",
		"xp_reward": 150,
		"require_employees": 5,
	},
	"tycoon": {
		"name": "Олигарх",
		"desc": "Нанять 9 сотрудников (максимум)",
		"xp_reward": 500,
		"require_employees": 9,
	},
	"technocrat": {
		"name": "Технократ",
		"desc": "Достичь максимального уровня системы",
		"xp_reward": 300,
		"require_system_level": 4,
	},
	"star_office": {
		"name": "Звёздный офис",
		"desc": "Достичь репутации 90+",
		"xp_reward": 200,
		"require_reputation": 90,
	},
	"queue_master": {
		"name": "Повелитель очереди",
		"desc": "Расширить очередь до 10 мест",
		"xp_reward": 100,
		"require_queue_size": 10,
	},
	"no_losses": {
		"name": "Без потерь",
		"desc": "Репутация 95+ и ни одного ушедшего",
		"xp_reward": 250,
		"require_reputation": 95,
		"require_zero_left": true,
	},
	"rich": {
		"name": "Богач",
		"desc": "Накопить 5000 XP",
		"xp_reward": 0,  # Награда — сам факт
		"require_xp": 5000,
	},
}

# Разблокированные достижения
var unlocked: Array[String] = []


## Проверить все достижения, вернуть список новых
func check_achievements(served_count: int, employee_count: int, system_level: int,
						reputation: int, left_count: int, queue_size: int, xp: int) -> Array:
	var new_unlocked = []

	for ach_id in Achievement:
		if unlocked.has(ach_id):
			continue  # Уже разблокировано

		var ach = Achievement[ach_id]
		if _check_condition(ach, served_count, employee_count, system_level,
							reputation, left_count, queue_size, xp):
			unlocked.append(ach_id)
			new_unlocked.append(ach_id)

	return new_unlocked


func _check_condition(ach: Dictionary, served: int, employees: int, system_level: int,
					  rep: int, left: int, queue: int, xp: int) -> bool:
	if ach.has("require_served") and served < ach.require_served:
		return false
	if ach.has("require_employees") and employees < ach.require_employees:
		return false
	if ach.has("require_system_level") and system_level < ach.require_system_level:
		return false
	if ach.has("require_reputation") and rep < ach.require_reputation:
		return false
	if ach.has("require_zero_left") and left > 0:
		return false
	if ach.has("require_queue_size") and queue < ach.require_queue_size:
		return false
	if ach.has("require_xp") and xp < ach.require_xp:
		return false
	return true


## Получить суммарную награду за список достижений
func get_total_reward(ach_ids: Array) -> int:
	var total = 0
	for id in ach_ids:
		if Achievement.has(id):
			total += Achievement[id].xp_reward
	return total


## Загрузить разблокированные достижения
func load_unlocked(unlocked_list: Array) -> void:
	unlocked.clear()
	for item in unlocked_list:
		unlocked.append(str(item))


## Получить список всех разблокированных
func get_unlocked() -> Array[String]:
	return unlocked


## Получить данные для UI
func get_achievement_data(ach_id: String) -> Dictionary:
	if Achievement.has(ach_id):
		var data = Achievement[ach_id].duplicate()
		data.id = ach_id
		data.unlocked = unlocked.has(ach_id)
		return data
	return {}
