import { createLazyFileRoute } from '@tanstack/react-router'
import hids_dashboard from '@/features/hids_dashboard'

export const Route = createLazyFileRoute('/_authenticated/hids_dashboard/')({
  component: hids_dashboard,
})
