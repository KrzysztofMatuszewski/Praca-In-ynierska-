import { createLazyFileRoute } from '@tanstack/react-router'
import training_panel from '@/features/training_panel'

export const Route = createLazyFileRoute('/_authenticated/training_panel/')({
  component: training_panel,
})
