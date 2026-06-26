-- Run this SQL in your Supabase SQL Editor to create the required tables.

-- Screening results
CREATE TABLE IF NOT EXISTS screenings (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users,
  test_type TEXT NOT NULL DEFAULT 'retinopathy',
  filename TEXT NOT NULL,
  image_b64 TEXT,
  primary_diagnosis TEXT NOT NULL,
  primary_confidence FLOAT NOT NULL,
  all_predictions JSONB NOT NULL,
  processing_time_ms FLOAT,
  device_info JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast user history queries
CREATE INDEX IF NOT EXISTS idx_screenings_user_id ON screenings (user_id);
CREATE INDEX IF NOT EXISTS idx_screenings_created_at ON screenings (created_at DESC);

-- Enable Row Level Security
ALTER TABLE screenings ENABLE ROW LEVEL SECURITY;

-- Users can only read their own screenings
CREATE POLICY "Users can read own screenings"
  ON screenings FOR SELECT
  USING (auth.uid() = user_id);

-- Users can insert their own screenings
CREATE POLICY "Users can insert own screenings"
  ON screenings FOR INSERT
  WITH CHECK (auth.uid() = user_id);
