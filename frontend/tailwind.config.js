/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', '"Inter"', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['"Plus Jakarta Sans"', '"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        navy: {
          950: '#060B19',
          900: '#0A1128',
          800: '#0F172A',
          700: '#1E293B',
          600: '#334155',
        },
        electric: {
          600: '#1D4ED8',
          500: '#2563EB',
          400: '#3B82F6',
          300: '#60A5FA',
          glow: '#06B6D4',
        },
      },
      boxShadow: {
        'premium': '0 4px 24px -4px rgba(10, 17, 40, 0.06), 0 2px 8px -2px rgba(10, 17, 40, 0.04)',
        'premium-lg': '0 12px 36px -6px rgba(10, 17, 40, 0.10), 0 4px 12px -2px rgba(37, 99, 235, 0.06)',
        'electric': '0 8px 24px -4px rgba(37, 99, 235, 0.35)',
      },
    },
  },
  plugins: [],
}

