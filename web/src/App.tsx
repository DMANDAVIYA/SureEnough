import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Footer from './components/Footer';
import Header from './components/Header';
import ViewBoundary from './components/ViewBoundary';
import { ManifestProvider, useManifest } from './data/manifest';
import Demo from './views/Demo';
import How from './views/How';
import Results from './views/Results';

function MockNotice() {
  return useManifest().mock ? (
    <p className="mock-notice" role="note">
      Synthetic mock data for development. These replays are simulated, not THINGS-EEG2 recordings.
    </p>
  ) : null;
}

export default function App() {
  const { pathname } = useLocation();
  return (
    <div className="app">
      <Header />
      <nav className="tabs" aria-label="Sections">
        <NavLink to="/" end>
          Demo
        </NavLink>
        <NavLink to="/results">Results</NavLink>
        <NavLink to="/how">How it works</NavLink>
      </nav>
      <main>
        <ManifestProvider>
          <MockNotice />
          <ViewBoundary key={pathname}>
            <Routes>
              <Route path="/" element={<Demo />} />
              <Route path="/results" element={<Results />} />
              <Route path="/how" element={<How />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ViewBoundary>
        </ManifestProvider>
      </main>
      <Footer />
    </div>
  );
}
