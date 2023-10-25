Query multiple tables at the same time.

The 2 tables share the same column.

The primary key and foreign key.
# INNER JOIN
Only return the rows which both have value in this column
```SQL
select * from player
inner join equip
on player.id = equip.player_id
```
# LEFT JOIN
Return all data of the left table, the right table will be filled with NULL if without value.
```SQL
select * from player
left join equip
on player.id = equip.player_id
```
If some player have the equipment, will show the rows from table `equip`.
Else, show NULL for each column in `equip`.
# RIGHT JOIN
reverse.
```SQL
select * from player
inner join equip
on player.id = equip.player_id
```