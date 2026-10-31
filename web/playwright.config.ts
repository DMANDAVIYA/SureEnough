import { defineConfig, devices } from '@playwright/test';

process.env.PLAYWRIGHT_BROWSERS_PATH ??= '0';

const PORT = 4174;

export default defineConfig({
  testDir: 'e2e',
  reporter: 'list',
  use: { baseURL: `http://localhost:${PORT}` },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `npm run mock && npm run build && npm run preview -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/SureEnough/`,
    env: { VITE_DATA_DIR: 'data/mock' },
    reuseExistingServer: false,
    timeout: 300_000,
  },
});
