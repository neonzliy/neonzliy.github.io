export function showFallback(stage,reason='static'){
  stage.dataset.state=reason;
  document.documentElement.classList.remove('has-webgl');
  const button=document.querySelector('[data-motion]');
  if(button){button.hidden=true;}
}
