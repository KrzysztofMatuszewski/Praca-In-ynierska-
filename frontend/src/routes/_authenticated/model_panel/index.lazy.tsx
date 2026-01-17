import { createLazyFileRoute } from '@tanstack/react-router'
import hids_dashboard from '@/features/model_panel'

export const Route = createLazyFileRoute('/_authenticated/model_panel/')({
  component: hids_dashboard,
})
