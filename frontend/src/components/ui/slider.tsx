import type { InputHTMLAttributes } from 'react'
import { cn } from '../../lib/utils'

export function Slider({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn('slider', className)} type="range" {...props} />
}
