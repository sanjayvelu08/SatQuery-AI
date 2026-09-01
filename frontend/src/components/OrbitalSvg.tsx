interface OrbitalSvgProps {
  size?: number;
  className?: string;
}

export function OrbitalSvg({ size = 80, className = '' }: OrbitalSvgProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 80 80"
      className={className}
      fill="none"
    >
      {/* Outer orbit */}
      <circle cx="40" cy="40" r="35" stroke="#1E3A5F" strokeWidth="1" />
      <circle
        cx="40"
        cy="40"
        r="35"
        stroke="#00D4AA"
        strokeWidth="1.5"
        strokeDasharray="20 200"
        strokeLinecap="round"
        className="animate-orbital-spin"
        style={{ transformOrigin: '40px 40px' }}
      />

      {/* Inner orbit */}
      <circle cx="40" cy="40" r="22" stroke="#1E3A5F" strokeWidth="0.5" />
      <circle
        cx="40"
        cy="40"
        r="22"
        stroke="#00E5FF"
        strokeWidth="1"
        strokeDasharray="10 130"
        strokeLinecap="round"
        className="animate-orbital-spin"
        style={{
          transformOrigin: '40px 40px',
          animationDuration: '6s',
          animationDirection: 'reverse',
        }}
      />

      {/* Earth/center dot */}
      <circle cx="40" cy="40" r="6" fill="#0A1628" stroke="#00D4AA" strokeWidth="1" />
      <circle cx="40" cy="40" r="3" fill="#00D4AA" opacity="0.3" />
      <circle cx="40" cy="40" r="1.5" fill="#00D4AA" />

      {/* Satellite dot on outer orbit */}
      <circle cx="40" cy="5" r="2" fill="#00E5FF" className="animate-orbital-spin" style={{ transformOrigin: '40px 40px' }} />
    </svg>
  );
}
