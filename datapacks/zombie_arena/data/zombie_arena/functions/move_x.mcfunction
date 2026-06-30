# Move the new zombie EAST (+X) by 2 blocks, repeated #ix times (recursive)
execute if score #ix arena matches 1.. as @e[type=zombie,tag=arena_new,limit=1] at @s run tp @s ~2 ~ ~
execute if score #ix arena matches 1.. run scoreboard players remove #ix arena 1
execute if score #ix arena matches 1.. run function zombie_arena:move_x
