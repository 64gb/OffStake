extends Node2D
## Слот декора — расставляется в редакторе
## decor_id задаётся в инспекторе: "plant", "cooler", "clock", "trash", "board"

@export var decor_id: String = "":
	set(value):
		decor_id = value

func is_purchased() -> bool:
	return false  # Заменится main.gd при инициализации

func mark_purchased() -> void:
	var indicator = get_node_or_null("SlotIndicator")
	var label = get_node_or_null("SlotLabel")
	if indicator:
		indicator.visible = false
	if label:
		label.visible = false
