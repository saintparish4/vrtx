/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  async rewrites() {
    return [
      {
        source: '/api/predict/:path*',
        destination: 'http://localhost:8000/predict/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
