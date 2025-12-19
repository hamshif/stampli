import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Helper: Stream Response
  const streamResponse = async (apiMessages) => {
    setLoading(true)
    try {
      const response = await fetch('/api/stampli', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: apiMessages })
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      // Add placeholder for assistant message
      setMessages(prev => [...prev, { role: 'assistant', content: '', logs: [] }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '')
            if (dataStr === '[DONE]') continue

            try {
              const event = JSON.parse(dataStr)

              setMessages(prev => {
                const newMsgs = [...prev]
                const lastIndex = newMsgs.length - 1
                const lastMsg = { ...newMsgs[lastIndex] }

                if (event.type === 'log') {
                  if (!lastMsg.logs) lastMsg.logs = []
                  lastMsg.logs = [...lastMsg.logs, event.data]
                } else if (event.type === 'delta') {
                  lastMsg.content += event.data.text
                }

                newMsgs[lastIndex] = lastMsg
                return newMsgs
              })
            } catch (e) {
              console.error("Parse error", e)
            }
          }
        }
      }
    } catch (err) {
      console.error("Fetch error", err)
      setMessages(prev => [...prev, { role: 'system', content: `Error: ${err.message}` }])
    } finally {
      setLoading(false)
    }
  }

  // Effect: Auto-Greeting
  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    const initGreeting = async () => {
      const triggerMsg = { role: 'user', content: 'Hi' }
      await streamResponse([triggerMsg])
    }
    initGreeting()
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMsg = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')

    const apiMessages = [...messages, userMsg].map(({ role, content }) => ({ role, content }))
    await streamResponse(apiMessages)
  }

  return (
    <div className="chat-container">
      <header>
        {/* The Digital Seal Logo */}
        <div className="logo-seal">印</div>
        <h1>Stampli / Agentic Analyst</h1>
      </header>

      <div className="messages-list">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            {msg.role === 'assistant' && msg.logs && msg.logs.length > 0 && (
              <div className="logs">
                {msg.logs.map((log, lIdx) => (
                  <div key={lIdx} className={`log-item ${log.status}`}>
                    <span style={{opacity: 0.5}}>[SYS]</span> {log.text}
                  </div>
                ))}
              </div>
            )}
            <div className="bubble">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        {loading && messages.length > 0 && !messages[messages.length - 1].content && (
          <div className="message assistant"><div className="bubble">...</div></div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Query the database..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'PROCESSING' : 'TRANSMIT'}
        </button>
      </form>
    </div>
  )
}

export default App