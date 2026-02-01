/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static optimization
  reactStrictMode: true,
  
  // Configure rewrites for API calls
  async rewrites() {
    return {
      beforeFiles: [
        {
          source: '/api/:path*',
          destination: `${process.env.NEXT_PUBLIC_API_URL}/api/:path*`,
        },
      ],
    }
  },

  // Configure headers for security
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin'
          }
        ]
      }
    ]
  },

  // Configure redirects
  async redirects() {
    return [
      {
        source: '/docs',
        destination: '/api/docs',
        permanent: false,
      },
    ]
  },

  // Enable image optimization (if using Next.js Image component)
  images: {
    domains: [],
    unoptimized: true, // Set to false in production if possible
  },

  // Environment variables
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
}

module.exports = nextConfig
