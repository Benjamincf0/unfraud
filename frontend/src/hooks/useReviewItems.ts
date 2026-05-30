import { useEffect, useState } from 'react'
import { loadReviewItems, type ReviewDataResult } from '../api/review'

type ReviewItemsState =
  | { status: 'loading'; data: null }
  | { status: 'ready'; data: ReviewDataResult }

export function useReviewItems() {
  const [state, setState] = useState<ReviewItemsState>({
    status: 'loading',
    data: null,
  })

  useEffect(() => {
    let isMounted = true

    loadReviewItems().then((data) => {
      if (isMounted) {
        setState({ status: 'ready', data })
      }
    })

    return () => {
      isMounted = false
    }
  }, [])

  return state
}
