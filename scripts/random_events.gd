## Случайные события
## Срабатывают каждые 2-5 минут, зависят от репутации

const Event = {
	# Хорошие события (репутация > 60)
	"inspection_passed": {
		"name": "Проверка пройдена!",
		"text": "Инспекция из министерства довольна! +20 XP",
		"min_rep": 60,
		"type": "good",
		"xp": 20,
		"rep": 5,
	},
	"minister_thanks": {
		"name": "Благодарность",
		"text": "Благодарность от министра! +15 XP",
		"min_rep": 70,
		"type": "good",
		"xp": 15,
		"rep": 3,
	},
	"bonus_day": {
		"name": "Премия!",
		"text": "Фонд развития выделил премию! +25 XP",
		"min_rep": 50,
		"type": "good",
		"xp": 25,
		"rep": 0,
	},
	"happy_visitor": {
		"name": "Отзыв",
		"text": "Посетитель оставил положительный отзыв! +10 XP, +3 реп.",
		"min_rep": 40,
		"type": "good",
		"xp": 10,
		"rep": 3,
	},

	# Плохие события (репутация < 40)
	"complaint": {
		"name": "Жалоба!",
		"text": "Жалоба в книгу жалоб! -5 репутации",
		"min_rep": 0,
		"max_rep": 40,
		"type": "bad",
		"xp": 0,
		"rep": -5,
	},
	"computer_crash": {
		"name": "Сбой системы!",
		"text": "Компьютер завис! Следующее обслуживание +30 сек",
		"min_rep": 0,
		"max_rep": 50,
		"type": "bad",
		"xp": 0,
		"rep": -3,
		"service_penalty": 30.0,
	},
	"audit": {
		"name": "Проверка!",
		"text": "Внеплановая проверка! Ускорьте работу! -8 реп.",
		"min_rep": 0,
		"max_rep": 35,
		"type": "bad",
		"xp": 0,
		"rep": -8,
	},

	# Нейтральные события (любая репутация)
	"normal_day": {
		"name": "Обычный день",
		"text": "Ничего особенного не происходит.",
		"min_rep": 0,
		"type": "neutral",
		"xp": 0,
		"rep": 0,
	},
	"coffee_break": {
		"name": "Перерыв",
		"text": "Сотрудники пьют кофе. Настроение хорошее! +3 XP",
		"min_rep": 0,
		"type": "neutral",
		"xp": 3,
		"rep": 1,
	},
	"new_equipment": {
		"name": "Оборудование",
		"text": "Привезли новый принтер! +5 XP",
		"min_rep": 0,
		"type": "neutral",
		"xp": 5,
		"rep": 0,
	},
}

# Таймер до следующего события
var event_timer: float = 0.0
var next_event_in: float = 120.0  # Первое событие через 2 мин

# Сигнал для UI
signal event_triggered(event_data: Dictionary)


func _process(delta: float, reputation: int, game_paused: bool) -> Dictionary:
	if game_paused:
		return {}

	event_timer += delta
	if event_timer < next_event_in:
		return {}

	# Сбрасываем таймер
	event_timer = 0.0
	next_event_in = randf_range(120.0, 300.0)  # 2-5 минут

	# Выбираем событие
	return _pick_event(reputation)


func _pick_event(reputation: int) -> Dictionary:
	var good_events = []
	var bad_events = []
	var neutral_events = []

	for event_id in Event:
		var event = Event[event_id]
		var rep = reputation

		if rep >= event.min_rep:
			if event.has("max_rep") and rep > event.max_rep:
				continue

			match event.type:
				"good": good_events.append(event)
				"bad": bad_events.append(event)
				"neutral": neutral_events.append(event)

	# Выбираем категорию по репутации
	var pool = []
	# Один бросок на выбор категории (раньше использовались повторные randf() в elif,
	# из-за чего реальные вероятности не совпадали с заявленными в комментариях)
	var roll = randf()
	if reputation >= 60:
		# 60% хорошие, 10% плохие, 30% нейтральные
		if roll < 0.60 and good_events.size() > 0:
			pool = good_events
		elif roll < 0.70 and bad_events.size() > 0:
			pool = bad_events
		elif neutral_events.size() > 0:
			pool = neutral_events
	elif reputation <= 40:
		# 15% хорошие, 55% плохие, 30% нейтральные
		if roll < 0.15 and good_events.size() > 0:
			pool = good_events
		elif roll < 0.70 and bad_events.size() > 0:
			pool = bad_events
		elif neutral_events.size() > 0:
			pool = neutral_events
	else:
		# Нейтральная зона: 25% хорошие, 25% плохие, 50% нейтральные
		if roll < 0.25 and good_events.size() > 0:
			pool = good_events
		elif roll < 0.50 and bad_events.size() > 0:
			pool = bad_events
		elif neutral_events.size() > 0:
			pool = neutral_events

	if pool.size() == 0:
		if neutral_events.size() > 0:
			pool = neutral_events
		else:
			return {}

	return pool[randi() % pool.size()]
