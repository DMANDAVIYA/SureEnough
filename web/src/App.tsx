import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import Footer from './components/Footer';
import Header from './components/Header';

function Placeholder({ title }: { title: string }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      <p>This view is being built.</p>
    </section>
  );
}

export default function App() {
  return (
    <div className="app">
      <Header />
      <nav className="tabs" aria-label="Sections">
        <NavLink to="/" end>Demo</NavLink>
        <NavLink to="/results">Results</NavLink>
        <NavLink to="/how">How it works</NavLink>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Placeholder title="Demo" />} />
          <Route path="/results" element={<Placeholder title="Results" />} />
          <Route path="/how" element={<Placeholder title="How it works" />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}
