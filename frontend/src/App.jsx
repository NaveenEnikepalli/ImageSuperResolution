import { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Enhance from './pages/Enhance';

export default function App() {
  const [activePage, setActivePage] = useState('Home');

  return (
    <div className="app-container">
      <Navbar activePage={activePage} setActivePage={setActivePage} />

      <main style={{ minHeight: '600px' }}>
        {activePage === 'Home' ? (
          <Home onStartEnhancing={() => setActivePage('Enhance')} />
        ) : (
          <Enhance />
        )}
      </main>

      <Footer />
    </div>
  );
}
