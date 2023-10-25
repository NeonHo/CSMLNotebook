```SQL
select * from player where level > (select AVG(level) from player);
```

```SQL
select level, ROUND((select AVG(level) from player)) from player;
```
- ROUND float to INT
```SQL
select level, ROUND((select AVG(level) from player)) as average,
level - ROUND((select AVG(level) from player))
```