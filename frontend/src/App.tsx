import { ReviewQueue } from './components/ReviewQueue'
import { mockReviewItems } from './data/mockReviewItems'

function App() {
  return <ReviewQueue items={mockReviewItems} />
}

export default App
