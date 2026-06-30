# === 0.5-second clock (called every game tick by the CLOCK command block) ===
scoreboard players add #t arena 1
execute if score #t arena matches 10.. run scoreboard players set #t arena 0
# every tick: the instant a target dies (Health 0), remove the corpse so its ~1s death animation never lingers
# on screen to be re-detected and re-shot (鞭尸). The arrow already registered the kill.
kill @e[type=zombie,tag=arena_zombie,nbt={Health:0.0f}]
# when the timer wrapped to 0 this tick AND the system is on -> run one spawn cycle (2x/sec = fast refill)
execute if score #t arena matches 0 if score #on arena matches 1 run function zombie_arena:tick
