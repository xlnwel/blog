# Optimizing React Hooks

React hooks like `useEffect`, `useCallback`, and `useMemo` are powerful tools, but they can be double-edged swords if not used correctly. In this article, we'll explore how to avoid common performance pitfalls.

## The Referential Equality Trap

One of the most common issues in React is **referential equality**. 

```javascript
// This function is recreated on every render
const handleClick = () => {
  console.log('Clicked');
};
```

If you pass `handleClick` to a child component wrapped in `React.memo`, that child will re-render every time the parent does, because the function is technically "new" every time.

### The Fix: useCallback

```javascript
const handleClick = useCallback(() => {
  console.log('Clicked');
}, []); // Dependencies
```

## When to use useMemo

Use `useMemo` for expensive calculations.

1. Large array filtering
2. Complex data transformation
3. Referential stability for objects passed to contexts

> **Pro Tip:** Don't optimize prematurely. React is fast. Only optimize when you see a bottleneck.

## Conclusion

Understanding the dependency array is key to mastering hooks. Always ensure your ESLint config includes `react-hooks/exhaustive-deps`.
