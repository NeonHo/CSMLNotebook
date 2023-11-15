# Type declaration
```TypeScript
let foo:string;
```
`string` is the type.

- If you use a variable, give it a value first to avoid `undefined` error.
- Type declaration is not necessary
	- TS will judge it by itself.
	- If the new value's type is inconsistent with the variable, error.
# TypeScript's compilation
TS -> JS for browser and Node.js


# Value & type

The type is an attribute which is belonged to the value.

The real code is JS to process the value.

The type code is TS to process the type.

During the compilation, type will be removed.

# tsconfig.json

arguments for complication of TS is written in this file.

# ts-node module

Using this, we can run the TS code directly.

The `npx` is calling the ts-node online, without intalling the ts-node module.
