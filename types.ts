import React from 'react';

export type ViewState = 'resume' | 'tech' | 'product' | 'business';

export interface Post {
  id: number;
  title: string;
  category: 'technical' | 'business';
  tags: string[];
  file: string;
  date: string;
  excerpt?: string; // Optional short summary
  readTime?: string;
}

export interface PostDetail extends Post {
  content: string;
}

export interface NavItem {
  id: ViewState;
  label: string;
  icon?: React.ReactNode;
}