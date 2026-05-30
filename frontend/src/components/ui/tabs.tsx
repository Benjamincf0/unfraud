import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '../../lib/utils'

type TabOption<T extends string> = {
  value: T
  label: ReactNode
}

type TabsProps<T extends string> = {
  value: T
  options: Array<TabOption<T>>
  onValueChange: (value: T) => void
  className?: string
}

export function Tabs<T extends string>({
  value,
  options,
  onValueChange,
  className,
}: TabsProps<T>) {
  return (
    <div className={cn('tabs', className)} role="tablist">
      {options.map((option) => (
        <TabButton
          aria-selected={value === option.value}
          key={option.value}
          onClick={() => onValueChange(option.value)}
          role="tab"
          type="button"
        >
          {option.label}
        </TabButton>
      ))}
    </div>
  )
}

function TabButton({
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={cn('tab-button', className)} {...props} />
}
