## Декор офиса — определения и логика

# Определения декора
var decor_definitions: Dictionary = {
	"plant": {
		"name": "[P] Растение",
		"desc": "Живое растение в горшке",
		"cost": 20,
		"color": Color(0.2, 0.7, 0.2, 1),
		"size": Vector2(30, 50)
	},
	"cooler": {
		"name": "[B] Кулер",
		"desc": "Кулер с водой",
		"cost": 30,
		"color": Color(0.3, 0.6, 0.9, 1),
		"size": Vector2(25, 55)
	},
	"clock": {
		"name": "[T] Часы",
		"desc": "Настенные часы",
		"cost": 25,
		"color": Color(0.9, 0.9, 0.9, 1),
		"size": Vector2(35, 35)
	},
	"trash": {
		"name": "[U] Урна",
		"desc": "Мусорная урна",
		"cost": 10,
		"color": Color(0.5, 0.5, 0.5, 1),
		"size": Vector2(25, 35)
	},
	"board": {
		"name": "[S] Стенд",
		"desc": "Информационный стенд",
		"cost": 40,
		"color": Color(0.8, 0.7, 0.4, 1),
		"size": Vector2(80, 30)
	}
}

# Купленный декор (id -> true/false)
var purchased_decor: Dictionary = {
	"plant": false,
	"cooler": false,
	"clock": false,
	"trash": false,
	"board": false,
}

func get_all_decor_ids() -> Array:
	return decor_definitions.keys()

func is_purchased(decor_id: String) -> bool:
	return purchased_decor.get(decor_id, false)

func purchase(decor_id: String, xp: int) -> Dictionary:
	"""Пытаемся купить декор. Возвращает {success: bool, xp_cost: int, message: String}"""
	if is_purchased(decor_id):
		return {"success": false, "xp_cost": 0, "message": "Уже куплено!"}
	
	var decor = decor_definitions.get(decor_id)
	if not decor:
		return {"success": false, "xp_cost": 0, "message": "Декор не найден!"}
	
	if xp < decor.cost:
		return {"success": false, "xp_cost": 0, "message": "Не хватает XP!"}
	
	purchased_decor[decor_id] = true
	return {"success": true, "xp_cost": decor.cost, "message": "Куплено: %s" % decor.name}

func get_decor_data_for_save() -> Dictionary:
	return purchased_decor.duplicate()

func load_decor_data(data: Dictionary) -> void:
	for key in data.keys():
		purchased_decor[key] = data[key]

# === БОНУСЫ ДЕКОРА ===
# +10 репутации за каждый элемент
# +2% XP за каждый элемент
# +15 сек терпения за каждый элемент

const REPUTATION_BONUS_PER_DECOR: int = 10
const XP_BONUS_PERCENT_PER_DECOR: float = 0.02
const PATIENCE_BONUS_SEC_PER_DECOR: float = 15.0

func get_purchased_count() -> int:
	var count = 0
	for decor_id in purchased_decor.keys():
		if purchased_decor[decor_id]:
			count += 1
	return count

func get_reputation_bonus() -> int:
	return get_purchased_count() * REPUTATION_BONUS_PER_DECOR

func get_xp_bonus_multiplier() -> float:
	return 1.0 + (get_purchased_count() * XP_BONUS_PERCENT_PER_DECOR)

func get_patience_bonus_seconds() -> float:
	return get_purchased_count() * PATIENCE_BONUS_SEC_PER_DECOR

func get_bonus_description() -> String:
	var count = get_purchased_count()
	if count == 0:
		return ""
	var parts: PackedStringArray = []
	var rep_bonus = get_reputation_bonus()
	if rep_bonus > 0:
		parts.append("[R] +%d репутации" % rep_bonus)
	var xp_bonus = int(get_xp_bonus_multiplier() * 100) - 100
	if xp_bonus > 0:
		parts.append("[X] +%d%% XP" % xp_bonus)
	var pat_bonus = get_patience_bonus_seconds()
	if pat_bonus > 0:
		parts.append("[T] +%d сек терпения" % pat_bonus)
	return " | ".join(parts)
