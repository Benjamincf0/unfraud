import type { HTMLAttributes } from 'react'
import { cn } from '../../lib/utils'

type BadgeTone = 'neutral' | 'critical' | 'high' | 'medium' | 'low' | 'success'

type BadgeProps = HTMLAttributes<HTMLSpanElement> & {
  tone?: BadgeTone
}

export function Badge({ className, tone = 'neutral', ...props }: BadgeProps) {
  return <span className={cn('badge', `badge-${tone}`, className)} {...props} />
}
