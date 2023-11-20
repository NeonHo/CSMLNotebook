如果你按照正常设置，
你的 tsconfig.json 中是有

    "baseUrl": ".",
    "paths": {
      "@/*": [
        "src/*"
      ]
    },

那么，我们的就只需要保证所有vue在第一级文件夹components中即可。
types文件夹与components