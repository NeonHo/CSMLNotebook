方便快速查询。
```SQL
CREATE [UNIQUE|FULLTEXT|SPATIAL] INDEX index_name ON tbl_name (index_col_name, ...)
```

例如
```SQL
create index email_index on fast (email)
show index from fast
select * from fast where email like 'abcd%' order by id
drop index email_index on fast
```

也可
```SQL
alter table fast add index name_index (name)
```
