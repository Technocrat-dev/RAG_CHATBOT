import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Required for Docker deployment - creates standalone output
  output: 'standalone',
  
  rewrites: async () => {
    const backendUrl = process.env.RAG_BACKEND_URL || 'http://127.0.0.1:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`,
      },
    ]
  },
  // Increase body parser limit for large file uploads
  experimental: {
    serverActions: {
      bodySizeLimit: '50mb',
    },
  },
};

export default nextConfig;