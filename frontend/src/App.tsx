import { ReviewQueue } from './components/ReviewQueue'
import { useReviewItems } from './hooks/useReviewItems'

function App() {
  const reviewItems = useReviewItems()

  if (reviewItems.status === 'loading') {
    return (
      <div className="loading-screen" role="status">
        Loading review queue
      </div>
    )
  }

  return (
    <ReviewQueue
      items={reviewItems.data.items}
      source={reviewItems.data.source}
    />
  )
}

export default App
