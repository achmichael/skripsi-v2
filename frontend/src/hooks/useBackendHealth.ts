import { useState, useEffect } from 'react'
import { api } from '@/lib/api'

type Status = 'checking' | 'online' | 'offline'

export function useBackendHealth() {
  const [status, setStatus] = useState<Status>('checking')

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        await api.health()
        if (!cancelled) setStatus('online')
      } catch {
        if (!cancelled) setStatus('offline')
      }
    }
    check()
    const interval = setInterval(check, 30000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return status
}
