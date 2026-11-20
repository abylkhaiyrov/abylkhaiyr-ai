export function Checkbox({ checked, disabled, label, onChange }: { checked: boolean; disabled?: boolean; label: string; onChange: () => void }) {
  return (
    <input
      type="checkbox"
      className="mt-1 h-4 w-4 shrink-0 cursor-pointer accent-[hsl(var(--primary))] disabled:cursor-not-allowed"
      checked={checked}
      disabled={disabled}
      aria-label={label}
      title={label}
      onChange={onChange}
    />
  );
}
