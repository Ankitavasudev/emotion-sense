import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Pie } from 'react-chartjs-2';
import { motion, AnimatePresence } from 'framer-motion';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

ChartJS.register(ArcElement, Tooltip, Legend);

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const EMOTION_CONFIG = {
  happy: { emoji: '😊', color: '#4CAF50', label: 'Happy' },
  sad: { emoji: '😢', color: '#2196F3', label: 'Sad' },
  angry: { emoji: '😠', color: '#f44336', label: 'Angry' },
  fear: { emoji: '😨', color: '#9C27B0', label: 'Fear' },
  surprise: { emoji: '😲', color: '#FF9800', label: 'Surprise' },
  neutral: { emoji: '😐', color: '#607D8B', label: 'Neutral' }
};

const EXAMPLE_TEXTS = [
  { text: "I absolutely love this product! It's amazing!", expected: "happy" },
  { text: "This is the worst experience I've ever had.", expected: "angry" },
  { text: "I'm worried about the upcoming exam.", expected: "fear" },
  { text: "Wow, I didn't expect that to happen!", expected: "surprise" },
  { text: "The meeting is scheduled for 3 PM.", expected: "neutral" },
  { text: "I feel so sad and lonely today.", expected: "sad" }
];

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetchHistory();
    fetchStats();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/history?limit=10`);
      setHistory(response.data.data || []);
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/stats`);
      setStats(response.data.data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const analyzeText = async () => {
    if (!text.trim()) {
      toast.warning('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/api/v1/analyze`, {
        text: text.trim()
      });

      setResult(response.data.data);
      toast.success('Analysis complete!');
      fetchHistory();
      fetchStats();
    } catch (error) {
      toast.error('Failed to analyze text. Please try again.');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const analyzeExample = (exampleText) => {
    setText(exampleText);
    analyzeText();
  };

  const getChartData = () => {
    if (!result) return null;

    const labels = Object.keys(result.scores).map(
      emotion => EMOTION_CONFIG[emotion]?.label || emotion
    );
    const data = Object.values(result.scores);
    const colors = Object.keys(result.scores).map(
      emotion => EMOTION_CONFIG[emotion]?.color || '#999'
    );

    return {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: '#fff'
      }]
    };
  };

  return (
    <div className="app">
      <ToastContainer position="top-right" autoClose={3000} />
      
      <div className="container">
        <motion.header 
          className="header"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1>🤖 Emotion Sense</h1>
          <p>Every prediction shows its reasons</p>
        </motion.header>

        <div className="main-content">
          <motion.div 
            className="input-section"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <h2>📝 Enter Text</h2>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type or paste text here..."
              rows={5}
            />
            
            <button 
              className="analyze-btn"
              onClick={analyzeText}
              disabled={loading}
            >
              {loading ? (
                <span className="loading-spinner"></span>
              ) : (
                '🔍 Analyze Sentiment'
              )}
            </button>

            <div className="examples">
              <h3>Try these:</h3>
              <div className="example-chips">
                {EXAMPLE_TEXTS.map((example, index) => (
                  <button
                    key={index}
                    className="chip"
                    onClick={() => analyzeExample(example.text)}
                  >
                    {EMOTION_CONFIG[example.expected].emoji} {example.text.substring(0, 30)}...
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          <AnimatePresence>
            {result && (
              <motion.div 
                className="result-section"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                <h2>📊 Analysis Result</h2>
                
                <div className="emotion-display">
                  <motion.div 
                    className="emotion-emoji"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 260, damping: 20 }}
                  >
                    {EMOTION_CONFIG[result.emotion]?.emoji || '❓'}
                  </motion.div>
                  <div className="emotion-name">
                    {EMOTION_CONFIG[result.emotion]?.label || result.emotion}
                  </div>
                  <div className="confidence">
                    {(result.confidence * 100).toFixed(1)}% confidence
                  </div>
                </div>

                <div className="scores-chart">
                  <Pie data={getChartData()} />
                </div>

                <div className="scores-list">
                  {Object.entries(result.scores).map(([emotion, score]) => (
                    <div key={emotion} className="score-item">
                      <span className="score-label">
                        {EMOTION_CONFIG[emotion]?.emoji} {EMOTION_CONFIG[emotion]?.label}
                      </span>
                      <div className="score-bar-container">
                        <motion.div 
                          className="score-bar"
                          initial={{ width: 0 }}
                          animate={{ width: `${score * 100}%` }}
                          style={{ backgroundColor: EMOTION_CONFIG[emotion]?.color }}
                        />
                      </div>
                      <span className="score-value">{(score * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {history.length > 0 && (
          <motion.div 
            className="history-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h2>📜 Recent History</h2>
            <div className="history-list">
              {history.map((item, index) => (
                <div key={index} className="history-item">
                  <span className="history-emoji">
                    {EMOTION_CONFIG[item.emotion]?.emoji}
                  </span>
                  <span className="history-text">
                    {item.text.substring(0, 50)}...
                  </span>
                  <span className="history-emotion">
                    {EMOTION_CONFIG[item.emotion]?.label}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        <footer className="footer">
          <p>Made with ❤️ by Ankita Salaria</p>
          <p className="tech-stack">FastAPI • React • TensorFlow</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
