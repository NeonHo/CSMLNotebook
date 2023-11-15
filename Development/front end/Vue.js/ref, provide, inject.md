
App.vue
```vue
<script setup lang="ts">
import Header from './components/Header.vue'

// step
const step = ref<string>("settings")  // reference variabel
provide('step',step)
</script>
```
./components/Header.vue
```vue
<script setup lang="ts">
import { Ref, inject } from 'vue'
// ref vars
const step = inject<Ref<string>>('step');
// functions
function change_step(newStepStr: string) {
    if (newStepStr == "analyze") {
        alert("功能优化中，暂未开放")
        return
    }
    console.log("change_step", newStepStr);
    if (step) {
        step.value = newStepStr
    }
}
</script>
```
