class_name Infomat
extends Area2D

## Инфомат - направляет посетителей к окнам

@export var detection_range: float = 50.0

func _ready() -> void:
	var collision = get_node_or_null("CollisionShape2D")
	if collision:
		collision.shape = CircleShape2D.new()
		collision.shape.radius = detection_range
