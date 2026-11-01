import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

// Mock axios
jest.mock('axios');

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
});
