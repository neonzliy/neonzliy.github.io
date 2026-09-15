export function getQuality(){
  const mobile=matchMedia('(max-width: 760px)').matches;
  return {mobile,dpr:Math.min(devicePixelRatio||1,mobile?1.25:1.65),reduced:matchMedia('(prefers-reduced-motion: reduce)').matches};
}
