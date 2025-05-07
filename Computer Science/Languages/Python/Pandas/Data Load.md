# Goal
Create [[Three Classes#DataFrame]] Object.
# From CSV
```Python
pd.read_csv(
	filepath,
	sep,  # 字段分隔符，默认逗号。
	header,  # Which row is the header? Specify it.
	encoding,  # 文件编码(Default: UTF-8)
	quotechar,  # ???
	usecols,  # ???
	index_col,  # Which column is the index column? Specify it.
	dtype,  # Sepecify the data type.
	converters,  # ???
	nrows,  # Number of rows.
	skiprows,  # Specify which rows you want to skip.
	parse_dates,  # Specify which columns you want to parse to dates.
	date_format, # Specify the date format.
	true_values,  # Specify which values will be regarded as the boolean True?
	false_values,  # Specify which values will be regarded as the boolean False?
	na_values,  # Specify which values will be regarded as Not A Number?
	na_filter,  # Whether filter the NaN mark?
	on_bad_lines,  # How to deal with bad lines? (error / warn / skip)
	engine,  # The engine "Arrow" can deal with lager dataset faster.
	iterator,  # If we turn this on, the less memory expense for more data.
	chunksize,  # The size loaded each time when the iterator is on.
)
```
1. `quotechar`：在 CSV 文件中，某些字段可能包含特殊字符，如逗号（`,` ），这些字符可能会被误解析为字段分隔符。使用引用字符可以明确地界定字段的边界。
```Python
import pandas as pd

data = 'col1,col2\n"value1, with comma",value2' 
with open('test.csv', 'w') as f:
	f.write(data)

df = pd.read_csv('test.csv') 
print(df)
```
2. `usecols` 参数用于指定要从 CSV 文件中读取哪些列。这在处理大型 CSV 文件且只需要部分列数据时非常有用，可以显著减少内存占用和读取时间。
```Python
import pandas as pd 
df = pd.read_csv('data.csv', usecols=['col1', 'col3'])
```

```Python
import pandas as pd 
df = pd.read_csv('data.csv', usecols=lambda x: len(x) > 3)
```
# From Excel
```Python
pd.read_excel(
	io,
	sheet_name,
	skip_footer,
)
```
# From SQL
