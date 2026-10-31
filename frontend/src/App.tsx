import { Header } from './components/layout/Header'
import { PredictionPage } from './features/predict/PredictionPage'
import { useBackendHealth } from './hooks/useBackendHealth'

export function App() {
  const backendStatus = useBackendHealth()

  return (
    <div className="min-h-screen flex flex-col">
      <Header backendStatus={backendStatus} />
      <PredictionPage />
    </div>
  )
}
