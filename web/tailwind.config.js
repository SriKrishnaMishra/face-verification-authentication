/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0f172a",
        card: "#111827",
        primary: "#2563eb",
        text: "#e2e8f0",
      },
    },
  },
  plugins: [],
}
