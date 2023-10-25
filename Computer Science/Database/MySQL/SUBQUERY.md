```SQL
select * from player where level > (select AVG(level) from player);
```

```SQL
select level, ROUND((select AVG(level) from player)) from player;
```
- ROUND float to INT
```SQL
select level, ROUND((select AVG(level) from player)) as average,
level - ROUND((select AVG(level) from player)) as diff
from player;
```
- `as` can name the new column for subquery.

# create new table from source table

```SQL
create table new_player select * from player where level < 5
```
# insert new rows into table
```SQL
insert into new_player select * from player where level between 6 and 10
```
# judge exist
```SQL
select exists ()
```