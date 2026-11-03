import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import App from './App';

// Mock axios
jest.mock('axios');

const ANALYSIS = {
  id: 'test-id',
  text: 'I am not happy with this result',
  emotion: 'sad',
  confidence: 0.498,
  scores: { happy: 0.1, sad: 0.498, angry: 0.1, fear: 0.1, surprise: 0.1, neutral: 0.102 },
  signals: [
    { token: 'happy', emotion: 'sad', source_emotion: 'happy', weight: 0.96, negated: true },
  ],
  engine: 'lexicon',
  timestamp: '2026-11-03T00:00:00Z',
};

beforeEach(() => {
  axios.get.mockResolvedValue({ data: { data: [] } });
  axios.post.mockResolvedValue({ data: { status: 'success', data: ANALYSIS } });
});

afterEach(() => {
  jest.clearAllMocks();
});

describe('App Component', () => {
  test('renders header', () => {
    render(<App />);
    const header = screen.getByText(/Emotion Sense/i);
    expect(header).toBeInTheDocument();
  });

  test('renders textarea', () => {
    render(<App />);
    const textarea = screen.getByPlaceholderText(/Type or paste text here/i);
    expect(textarea).toBeInTheDocument();
  });

  test('renders analyze button', () => {
    render(<App />);
    const button = screen.getByText(/Analyze Sentiment/i);
    expect(button).toBeInTheDocument();
  });

  test('renders example chips', () => {
    render(<App />);
    const chips = screen.getAllByRole('button');
    expect(chips.length).toBeGreaterThan(1);
  });

  test('shows the words behind a prediction', async () => {
    render(<App />);
    fireEvent.change(screen.getByPlaceholderText(/Type or paste text here/i), {
      target: { value: 'I am not happy with this result' },
    });
    fireEvent.click(screen.getByText(/Analyze Sentiment/i));

    expect(await screen.findByText(/Why this result/i)).toBeInTheDocument();
    expect(await screen.findByText('happy')).toBeInTheDocument();
    expect(await screen.findByText(/negated from/i)).toBeInTheDocument();
    expect(await screen.findByText(/lexicon engine/i)).toBeInTheDocument();
  });

  test('an example chip analyzes the example, not the previous box value', async () => {
    render(<App />);
    const chip = screen
      .getAllByRole('button')
      .find((button) => button.textContent.includes('I absolutely love'));

    fireEvent.click(chip);

    await waitFor(() =>
      expect(axios.post).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/analyze'),
        { text: "I absolutely love this product! It's amazing!" }
      )
    );
  });
});
