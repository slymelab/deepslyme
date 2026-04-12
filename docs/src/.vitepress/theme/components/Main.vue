<template>
  <div class="relative min-h-screen overflow-hidden font-sans text-gray-900 dark:text-gray-50">
    <header
      class="relative z-10 w-full max-w-[1200px] mx-auto px-8 pt-20 pb-24 flex flex-col items-center text-center">
      <div class="flex flex-col items-center gap-8">
        <div class="flex items-center justify-center gap-4">
          <SlymeIcon char="D" size="lg" />
          <h1 class="font-mono text-6xl tracking-tighter">DeepSlyme</h1>
        </div>
        <p class="text-xl leading-relaxed font-medium max-w-5xl">
          DeepSlyme is a powerful LLM training framework built natively on Slyme.<br />
          <span class="text-gray-500 dark:text-gray-400 block mt-2 font-mono">Seamlessly orchestrate distributed training and accelerate model optimization.</span>
        </p>
        <div class="flex flex-col sm:flex-row items-center justify-center gap-8 w-full mt-2">
          <SlymeCodeSnippet command="pip install deepslyme" />
          <SlymeButtonLink :href="withBase('/guide/intro/what-is-deepslyme')" variant="outline">
            Get Started -&gt;
          </SlymeButtonLink>
        </div>
      </div>
    </header>
    <div class="w-full max-w-[1200px] mx-auto px-8">
      <hr class="border-gray-200/80">
    </div>

    <section class="w-full max-w-[1200px] mx-auto px-8 py-16">
      <div class="mb-12">
        <h2 class="font-mono text-3xl font-semibold tracking-tight">Why build with DeepSlyme?</h2>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <SlymeDescription>
          <template #title>Transparent Pipelines</template>
          <template #description>Escape the black box of traditional trainers. DeepSlyme eliminates tangled object hierarchies and endless code jumping, giving you absolute clarity and control over your entire training loop.</template>
        </SlymeDescription>
        <SlymeDescription>
          <template #title>High Extensibility</template>
          <template #description>Focus on algorithms rather than fighting boilerplate. Our decoupled architecture makes extending, debugging, and testing new ideas significantly faster and completely frictionless.</template>
        </SlymeDescription>
        <SlymeDescription>
          <template #title>Universal Composability</template>
          <template #description>Extract and reuse custom nodes across projects seamlessly. Integrate your breakthroughs into the broader community, and leverage out-of-the-box components from the entire Slyme ecosystem.</template>
        </SlymeDescription>
      </div>
    </section>
    <SlymeFooter>
      <template #brand>
        <div class="flex flex-row items-center">
          <SlymeIcon size="md" char="S" />
          <p class="ml-3 text-lg font-bold">Slyme Lab</p>
        </div>
        <p class="font-mono text-sm leading-relaxed text-gray-600 dark:text-gray-400">
          A unified ecosystem for building, orchestrating, and scaling functional pipelines with ultimate simplicity.
        </p>
      </template>
      <template #columns>
        <SlymeFooterColumn title="Resources">
          <li>
            <SlymeLink :href="withBase('/')" theme="muted" class="text-sm">Home</SlymeLink>
          </li>
          <li>
            <SlymeLink :href="withBase('/guide/intro/what-is-deepslyme')" theme="muted" class="text-sm">
              Guide
            </SlymeLink>
          </li>
        </SlymeFooterColumn>
        <SlymeFooterColumn title="Social Links">
          <li>
            <SlymeLink href="https://github.com/slymelab/deepslyme" target="_blank" theme="muted" class="text-sm">
              <div class="flex flex-row items-center">
                <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path
                    d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.09-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.63-.33 2.47-.33.84 0 1.68.11 2.47.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z" />
                </svg>
                <p class="text-sm ml-1">GitHub</p>
              </div>
            </SlymeLink>
          </li>
        </SlymeFooterColumn>
      </template>
      <template #bottom-left>
        &copy; {{ new Date().getFullYear() }} The SlymeLab Team.
      </template>
      <template #bottom-right>
        Built with Pythonic Passion.
      </template>
    </SlymeFooter>
  </div>
</template>

<script setup>
import { withBase } from 'vitepress'
import { SlymeDescription, SlymeCodeSnippet, SlymeButtonLink, SlymeIcon, SlymeFooter, SlymeFooterColumn, SlymeLink } from '@slymelab/theme'
</script>
