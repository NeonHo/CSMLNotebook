
App.vue
```vue
<script setup lang="ts">
import Header from './components/Header.vue'

// step
const step = ref<string>("settings")  // reference variabel
provide('step',step)
<script>
```
./components/Header.vue
```vue
<script setup lang="ts">
import { Ref, inject } from 'vue'
</script>
```
