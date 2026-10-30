import { HOOK, LABEL } from '../copy';

export default function Header() {
  return (
    <header className="site-header">
      <h1>SureEnough</h1>
      <p className="hook">{HOOK}</p>
      <p className="label" role="note">{LABEL}</p>
    </header>
  );
}
