/**
 * supabase.js — Integração com Supabase para salvar anotações remotamente.
 * Chave anon (somente INSERT, sem leitura — segura para uso público).
 */

const SUPA_URL = "https://rynxhuistciljkuqqcsh.supabase.co";
const SUPA_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5bnhodWlzdGNpbGprdXFxY3NoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUzNzI2NDAsImV4cCI6MjA5MDk0ODY0MH0.Dyu9D6mFg7ZGDgruPtWwcUBui1dABCIfKD6GLXWvciw";

const supa = supabase.createClient(SUPA_URL, SUPA_KEY);

async function saveToSupabase(ann) {
  const { error } = await supa.from("annotations").insert([ann]);
  if (error) {
    console.warn("Supabase error:", error.message);
    return false;
  }
  return true;
}
